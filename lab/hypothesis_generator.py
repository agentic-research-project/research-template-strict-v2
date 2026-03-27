"""
Stage 3: 가설 생성

[단일 모드] Claude 단독 생성
  python -m lab.hypothesis_generator \
    --topic-file  experiments/{slug}/reports/topic_analysis.json \
    --papers-file experiments/{slug}/reports/papers.json

[협업 모드] 5라운드 토론 기반 생성 (권장)
  python -m lab.hypothesis_generator \
    --topic-file  experiments/{slug}/reports/topic_analysis.json \
    --papers-file experiments/{slug}/reports/papers.json \
    --mode collaborative

협업 흐름:
  Round 0 — Gemini (Structurer) : 논문 → Evidence Pack 구조화
  Round 1 — Claude (Proposer)   : Evidence Pack 기반 가설 3개 제안
  Round 2 — GPT   (Critic)      : 논리 결함·과장·검증 불가 요소 공격적 비판
  Round 3 — Gemini (Mediator)   : 두 안 통합 + 검증 가능한 합성 가설 도출
  Round 4 — Claude (Finalizer)  : 최종 가설 + 실험 계획 + 반증 조건 확정
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from lab.config import (
    OPENAI_MODEL, GEMINI_MODEL,
    query_claude, get_openai_client, get_gemini_model, parse_json,
    topic_slug as _topic_slug,
)


# ──────────────────────────────────────────────────────────
# 공통 스키마
# ──────────────────────────────────────────────────────────

_HYPOTHESIS_SCHEMA = """{
  "research_gap_summary": "연구 갭 요약",
  "statement": "가설 (영어 1문장)",
  "statement_kr": "가설 (한국어 1문장)",
  "rationale": "가설 근거 (3-5문장)",
  "key_innovation": "핵심 혁신 포인트",
  "expected_mechanism": "예상 작동 원리",
  "falsification_criteria": ["반증 조건 1: 어떤 결과가 나오면 가설 기각", "반증 조건 2"],
  "architecture": "제안 모델 구조",
  "dataset": "데이터셋 제안",
  "baseline_models": ["기준 모델 1", "기준 모델 2"],
  "evaluation_metrics": ["지표 1", "지표 2"],
  "key_experiments": ["핵심 실험 1", "핵심 실험 2"],
  "ablation_studies": ["ablation 1", "ablation 2"],
  "related_papers": [
    {"paper_id": "paper-id", "title": "논문 제목", "venue": "학회", "relevance": "관련성"}
  ],
  "evidence_links": [
    {"paper_id": "paper-id", "supports": ["key_innovation", "expected_mechanism"]}
  ],
  "confidence": 0.0,
  "risk_factors": ["위험 1", "위험 2"]
}"""


# config.py 공통 상수 사용
from lab.config import EVIDENCE_COVERAGE_SLOTS as _MUST_COVER_SLOTS, COVERAGE_GROUPS


def _load_inputs(topic_file: str, papers_file: str):
    topic_data  = json.loads(Path(topic_file).read_text(encoding="utf-8"))
    papers_data = json.loads(Path(papers_file).read_text(encoding="utf-8"))
    topic_name  = topic_data.get("input", {}).get("topic", "research")
    topic_slug  = _topic_slug(topic_name)
    return topic_data, papers_data, topic_name, topic_slug


# ──────────────────────────────────────────────────────────
# B-6: Coverage Check Gate
# ──────────────────────────────────────────────────────────

def _coverage_check_gate(papers_data: dict, topic_data: dict) -> dict:
    """가설 생성 전 evidence coverage를 검사한다 (B-6).

    근거가 부족한 상태에서 가설을 만들지 않도록 gate를 적용한다.

    Returns:
        {
            "pass": True/False,
            "evidence_coverage": {...},
            "missing_critical": [...],
            "recommendation": "proceed" | "re_query" | "fail_fast",
            "suggested_queries": [...],
        }
    """
    evidence_coverage = papers_data.get("evidence_coverage", {})
    coverage = evidence_coverage.get("coverage", {})
    missing = evidence_coverage.get("missing_slots", [])
    papers = papers_data.get("papers", [])

    # coverage가 papers.json에 없으면 직접 계산
    if not coverage and papers:
        retrieval_plan = topic_data.get("retrieval_plan", {})
        must_cover = retrieval_plan.get("must_cover", _MUST_COVER_SLOTS)

        slot_counts = {slot: 0 for slot in must_cover}
        for p in papers:
            for slot in p.get("claim_slots_supported", []):
                if slot in slot_counts:
                    slot_counts[slot] += 1
        coverage = {slot: count > 0 for slot, count in slot_counts.items()}
        missing = [slot for slot, covered in coverage.items() if not covered]

        # closest_prior_art 역할 존재 여부
        has_prior_art = any(p.get("evidence_role") == "closest_prior_art" for p in papers)
        if not has_prior_art:
            coverage["closest_prior_art"] = False
            if "closest_prior_art" not in missing:
                missing.append("closest_prior_art")

    # slot-complete 판정: 3개 group별 최소 coverage 필요 (config.py 공통 계약)
    novelty_slots = COVERAGE_GROUPS["novelty"]
    validity_slots = COVERAGE_GROUPS["validity"]
    feasibility_slots = COVERAGE_GROUPS["feasibility"]

    novelty_ok = any(coverage.get(s, False) for s in novelty_slots)
    validity_ok = any(coverage.get(s, False) for s in validity_slots)
    feasibility_ok = any(coverage.get(s, False) for s in feasibility_slots)

    all_critical = novelty_slots | validity_slots | feasibility_slots
    critical_missing = [s for s in missing if s in all_critical]

    # 논문 수 자체가 너무 적으면 fail_fast
    if len(papers) < 3:
        return {
            "pass": False,
            "evidence_coverage": coverage,
            "missing_critical": critical_missing or missing,
            "recommendation": "fail_fast",
            "suggested_queries": [],
            "reason": f"논문 {len(papers)}편으로 근거 부족 (최소 3편 필요)",
            "group_coverage": {"novelty_ok": novelty_ok, "validity_ok": validity_ok, "feasibility_ok": feasibility_ok},
        }

    # group 중 2개 이상 미충족 → re_query
    failed_groups = []
    if not novelty_ok:
        failed_groups.append("novelty")
    if not validity_ok:
        failed_groups.append("validity")
    if not feasibility_ok:
        failed_groups.append("feasibility")

    if len(failed_groups) >= 2:
        inp = topic_data.get("input", {})
        topic_tokens = [t for t in re.split(r"\W+", inp.get("topic", "")) if len(t) > 3][:2]
        base = " ".join(topic_tokens)
        suggested = []
        if "novelty" in failed_groups:
            suggested.append(f"recent survey {base}")
            suggested.append(f"novel approach mechanism {base}")
        if "validity" in failed_groups:
            suggested.append(f"state-of-the-art benchmark {base}")
            suggested.append(f"evaluation metrics {base}")
        if "feasibility" in failed_groups:
            suggested.append(f"constraint-aware {base} deployment")
        return {
            "pass": False,
            "evidence_coverage": coverage,
            "missing_critical": critical_missing,
            "recommendation": "re_query",
            "suggested_queries": suggested,
            "reason": f"group coverage 부족: {failed_groups}",
            "group_coverage": {"novelty_ok": novelty_ok, "validity_ok": validity_ok, "feasibility_ok": feasibility_ok},
        }

    # 1개 group만 부족 → proceed with warning
    if failed_groups:
        return {
            "pass": True,
            "evidence_coverage": coverage,
            "missing_critical": critical_missing,
            "recommendation": "proceed",
            "suggested_queries": [],
            "reason": f"group coverage warning: {failed_groups} 부족하지만 진행 가능",
            "group_coverage": {"novelty_ok": novelty_ok, "validity_ok": validity_ok, "feasibility_ok": feasibility_ok},
        }

    # 완전 통과
    return {
        "pass": True,
        "evidence_coverage": coverage,
        "missing_critical": [],
        "recommendation": "proceed",
        "suggested_queries": [],
        "group_coverage": {"novelty_ok": novelty_ok, "validity_ok": validity_ok, "feasibility_ok": feasibility_ok},
    }


def _build_context(input_info: dict, research_q: str, success_criteria: dict) -> str:
    """모든 라운드가 공통으로 받는 연구 컨텍스트 (evidence pack 제외)."""
    return f"""## 연구 주제
- 주제: {input_info.get('topic', '')}
- 문제 정의: {input_info.get('problem_definition', '')}
- 원하는 결과: {input_info.get('desired_outcome', '')}
- 제약 조건: {input_info.get('constraints', '')}
- 목표 지표: {input_info.get('target_metric', '')}

## 핵심 연구 질문
{research_q}

## 성공 기준
{json.dumps(success_criteria, ensure_ascii=False)}"""


def _print_round(n: int, label: str, statement: str,
                 note: str = "", critique: str = "") -> None:
    print(f"\n  {'─'*55}")
    print(f"  Round {n} — {label}")
    print(f"  {'─'*55}")
    if critique:
        print(f"  비판: {critique[:120]}")
    if note:
        print(f"  메모: {note[:120]}")
    print(f"  가설: {statement[:120]}")


# ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────
# paper_id 정규화 헬퍼
# ──────────────────────────────────────────────────────────

def _normalize_pid(raw_paper_id, raw_id, original_ids: list[str], i: int) -> str:
    """LLM이 반환한 paper_id/id를 안정적인 원본 ID로 복원한다.

    우선순위:
      1. raw_paper_id / raw_id 중 숫자 문자열이면 → 1-based 인덱스로 original_ids 매핑
      2. raw_paper_id가 비숫자 비어있지 않은 문자열 → 그대로 사용
      3. 위치 기반 fallback: original_ids[i]
      4. raw_id 문자열 그대로
      5. P{i+1}
    """
    def _try_index(val) -> str | None:
        if val is None:
            return None
        s = str(val).strip()
        if s.isdigit():
            idx = int(s) - 1
            if 0 <= idx < len(original_ids):
                return original_ids[idx]
            return original_ids[i] if i < len(original_ids) else f"P{i + 1}"
        return None

    # 두 필드 모두 숫자 매핑 시도 (raw_paper_id 우선)
    for candidate in [raw_paper_id, raw_id]:
        mapped = _try_index(candidate)
        if mapped:
            return mapped

    # raw_paper_id가 비숫자 유효 문자열이면 신뢰
    if raw_paper_id is not None:
        s = str(raw_paper_id).strip()
        if s:
            return s

    # 위치 기반 fallback
    if i < len(original_ids):
        return original_ids[i]

    if raw_id is not None:
        return str(raw_id).strip()

    return f"P{i + 1}"


# Round 0: Evidence Pack 구조화 (Gemini)
# ──────────────────────────────────────────────────────────

def _build_evidence_pack(papers: list[dict]) -> str:
    """
    각 논문을 '문제/방법/한계/실험조건/남는 빈틈' 포맷으로 구조화.
    모든 라운드가 동일한 입력을 받아 '해석 차이'가 아닌 '논리 차이'로 토론하게 한다.
    """
    model = get_gemini_model()

    def _paper_payload(idx: int, p: dict) -> dict:
        entry = {
            "paper_id": p.get("paper_id", f"P{idx}"),
            "title":    p.get("title", ""),
            "year":     p.get("year", ""),
            "abstract": (p.get("abstract") or "")[:1200],
            "url":      p.get("url", ""),
            "source":   p.get("source", ""),
        }
        if p.get("sections"):
            entry["sections"] = p["sections"]  # introduction/method/experiment/limitation
        return entry

    papers_raw = json.dumps(
        [_paper_payload(i + 1, p) for i, p in enumerate(papers[:15])],
        ensure_ascii=False, indent=2,
    )

    prompt = f"""당신은 딥러닝 분야 전문 연구 분석가입니다.
아래 논문들을 각각 동일한 구조적 포맷으로 분석하여 Evidence Pack을 작성하세요.
모든 논문이 같은 틀로 분석되어야 나중에 연구 갭과 가설을 비교할 수 있습니다.

⚠️ 논문에 'sections' 필드가 있으면 abstract 대신 그 섹션 텍스트를 우선 참조하여 더 깊은 분석을 하세요.
각 분석 항목은 가능하면 paper_id 기준으로 추적 가능해야 한다.

## 논문 목록
{papers_raw}

## 출력 형식 (JSON 배열)
반드시 아래 JSON 배열로만 답변하세요 (마크다운 없이):
[
  {{
    "paper_id": "arxiv:2401.12345",
    "title": "논문 제목",
    "problem": "이 논문이 해결하려는 핵심 문제 (1-2문장)",
    "method": "핵심 방법론 (50자 이내, 구체적 기술명 포함)",
    "limitations": "이 방법론의 핵심 한계 (1-2문장)",
    "experimental_conditions": "데이터셋 / 주요 지표 / 베이스라인",
    "remaining_gaps": "이 논문 이후에도 해결 못 한 핵심 미해결 문제"
  }}
]"""

    print("  [Round 0 / Gemini] Evidence Pack 구조화 중...")
    response = model.generate_content(prompt)

    # 입력 논문의 원본 paper_id 목록 (normalize 시 fallback용)
    original_ids = [
        p.get("paper_id", f"P{i + 1}")
        for i, p in enumerate(papers[:15])
    ]

    try:
        raw_list = parse_json(response.text)

        # paper_id normalize: _normalize_pid()로 "1","2" 숫자도 원본 ID로 복원
        pack_list = []
        for i, item in enumerate(raw_list):
            raw_paper_id = item.get("paper_id")
            raw_id       = item.get("id")
            pid = _normalize_pid(raw_paper_id, raw_id, original_ids, i)

            # B-4: 원본 논문에서 evidence 메타데이터 가져오기
            source_paper = papers[i] if i < len(papers) else {}

            pack_list.append({
                "paper_id":               str(pid),
                "title":                  item.get("title", ""),
                "source":                 source_paper.get("source", ""),
                "problem":                item.get("problem", ""),
                "method":                 item.get("method", ""),
                "limitations":            item.get("limitations", ""),
                "experimental_conditions": item.get("experimental_conditions", ""),
                "remaining_gaps":         item.get("remaining_gaps", ""),
                # B-3/B-4: evidence graph 필드 (논문에서 계승)
                "evidence_role":          source_paper.get("evidence_role", ""),
                "claim_slots_supported":  source_paper.get("claim_slots_supported", []),
                "support_strength":       source_paper.get("support_strength", "contextual"),
                "why_selected":           source_paper.get("why_selected", ""),
                "rank":                   i + 1,  # B-4: 순위
            })

        lines = ["## Evidence Pack — 논문 구조화 분석\n"]
        for p in pack_list:
            lines.append(f"[{p['paper_id']}] {p['title']}")
            lines.append(f"  ▶ 문제        : {p['problem']}")
            lines.append(f"  ▶ 방법        : {p['method']}")
            lines.append(f"  ▶ 한계        : {p['limitations']}")
            lines.append(f"  ▶ 실험 조건   : {p['experimental_conditions']}")
            lines.append(f"  ▶ 남는 빈틈   : {p['remaining_gaps']}")
            lines.append("")
        pack_str = "\n".join(lines)
        print(f"  → Evidence Pack 완성: {len(pack_list)}편 구조화")
        return pack_str, pack_list
    except Exception as e:
        print(f"  [경고] Evidence Pack JSON 파싱 실패: {e}")
        fallback_lines = ["## Evidence Pack — fallback\n"]
        for i, p in enumerate(papers[:15]):
            fallback_lines.append(
                f"[{p.get('paper_id', f'P{i+1}')}] {p.get('title', '')} :: {(p.get('abstract') or '')[:200]}"
            )
        return "\n".join(fallback_lines), []


# ──────────────────────────────────────────────────────────
# Round 1: Claude Proposer (가설 3개)
# ──────────────────────────────────────────────────────────

def _claude_propose(context: str, evidence_pack: str) -> dict:
    """Evidence Pack을 기반으로 가설 3개를 제안. GPT 비판 전 초안."""
    prompt = f"""당신은 딥러닝 연구 전문가입니다.
아래 Evidence Pack(논문 구조화 분석)과 연구 맥락을 바탕으로
가설 3개를 제안하세요. 각 가설은 서로 다른 접근법을 취해야 합니다.

{context}

{evidence_pack}

## 임무
Evidence Pack의 '남는 빈틈'을 분석하여 해결 가능한 연구 갭을 찾고,
서로 다른 관점에서 가설 3개를 제안하세요.
각 가설은 논문들과 구체적으로 무엇이 다른지 명시해야 합니다.

⚠️ **중요**: Evidence Pack에서 명확한 연구 갭을 찾지 못하거나,
논문 근거가 부족하여 신뢰할 수 있는 가설을 세울 수 없다면,
무리하게 가설을 생성하지 마세요. 대신 아래 형식으로 답변하세요:
{{"status": "insufficient_evidence", "reason": "구체적 사유", "suggestions": ["추가 검색 키워드 1", "키워드 2"]}}

## 출력 형식 (JSON만)
{{
  "research_gap_summary": "Evidence Pack에서 발견한 핵심 연구 갭",
  "existing_approaches": ["기존 접근법 1", "기존 접근법 2"],
  "limitations": ["공통 한계 1", "공통 한계 2"],
  "hypotheses": [
    {{
      "id": 1,
      "statement": "가설 1 (영어 1문장)",
      "statement_kr": "가설 1 (한국어 1문장)",
      "core_idea": "핵심 아이디어 (2-3문장)",
      "key_innovation": "기존 논문 대비 구체적 차별점",
      "potential_weaknesses": ["예상 약점 1", "약점 2"]
    }},
    {{
      "id": 2,
      "statement": "가설 2 (영어 1문장)",
      "statement_kr": "가설 2 (한국어 1문장)",
      "core_idea": "핵심 아이디어 (2-3문장)",
      "key_innovation": "기존 논문 대비 구체적 차별점",
      "potential_weaknesses": ["예상 약점 1", "약점 2"]
    }},
    {{
      "id": 3,
      "statement": "가설 3 (영어 1문장)",
      "statement_kr": "가설 3 (한국어 1문장)",
      "core_idea": "핵심 아이디어 (2-3문장)",
      "key_innovation": "기존 논문 대비 구체적 차별점",
      "potential_weaknesses": ["예상 약점 1", "약점 2"]
    }}
  ]
}}"""

    return parse_json(query_claude(prompt))


# ──────────────────────────────────────────────────────────
# Round 2: GPT Critic (3개 가설 공격적 비판)
# ──────────────────────────────────────────────────────────

def _gpt_critique(context: str, evidence_pack: str, claude_draft: dict) -> dict:
    """GPT가 3개 가설 각각의 논리 결함·과장·검증 불가 요소를 비판."""
    client = get_openai_client()

    prompt = f"""당신은 엄격한 딥러닝 연구 리뷰어입니다.
Claude가 제안한 가설 3개를 Evidence Pack을 기반으로 비판적으로 검토하세요.
"논리 결함", "과장", "검증 불가능한 주장"에 집중하세요.
긍정적 평가보다 약점 발굴에 집중하세요.

{context}

{evidence_pack}

## Claude의 가설 3개
{json.dumps(claude_draft.get('hypotheses', []), ensure_ascii=False, indent=2)}

## 비판 지침
- Evidence Pack에 근거하여 "이미 해결된 문제" 또는 "선행 연구와 실질 차이 없음"을 지적
- 검증에 필요한 데이터·환경이 현실적으로 불가능한 경우 지적
- 과도한 성능 목표나 근거 없는 주장 지적
- 각 가설 중 가장 유망한 것을 선택하고 그 이유 설명

## 출력 형식 (JSON만)
{{
  "critiques": [
    {{
      "id": 1,
      "fatal_flaws": ["치명적 결함 1", "결함 2"],
      "exaggerations": ["과장된 주장 1"],
      "unverifiable": ["검증 불가 요소 1"],
      "score": 0
    }},
    {{
      "id": 2,
      "fatal_flaws": ["치명적 결함 1"],
      "exaggerations": [],
      "unverifiable": ["검증 불가 요소 1"],
      "score": 0
    }},
    {{
      "id": 3,
      "fatal_flaws": ["치명적 결함 1"],
      "exaggerations": ["과장된 주장 1"],
      "unverifiable": [],
      "score": 0
    }}
  ],
  "best_hypothesis_id": 1,
  "best_hypothesis_reason": "선택 이유",
  "critique_summary": "3개 가설의 공통 약점 요약",
  "missed_opportunities": ["Claude가 놓친 연구 갭 1", "놓친 관점 2"]
}}"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ──────────────────────────────────────────────────────────
# Round 2.5: Pairwise Ranking (GPT + Gemini 쌍대비교)
# ──────────────────────────────────────────────────────────

def _pairwise_rank(context: str, hypotheses: list[dict]) -> dict:
    """3개 가설에 대해 GPT + Gemini가 pairwise 비교하여 순위를 매긴다.

    절대점수는 gate (최소 기준), pairwise는 selection (최종 선택)으로 사용.
    비교 쌍: (1 vs 2), (1 vs 3), (2 vs 3)
    """
    if len(hypotheses) < 2:
        return {"ranking": [h.get("id", i+1) for i, h in enumerate(hypotheses)],
                "pairwise_results": []}

    pairs = []
    for i in range(len(hypotheses)):
        for j in range(i + 1, len(hypotheses)):
            pairs.append((hypotheses[i], hypotheses[j]))

    pair_prompt_parts = []
    for a, b in pairs:
        pair_prompt_parts.append(
            f"### 가설 {a.get('id')} vs 가설 {b.get('id')}\n"
            f"  A: {a.get('statement_kr', a.get('statement', ''))}\n"
            f"  B: {b.get('statement_kr', b.get('statement', ''))}"
        )

    prompt = f"""당신은 딥러닝 연구 전문가입니다.
아래 가설 쌍들을 비교하여 각 쌍에서 더 나은 가설을 선택하세요.

{context}

## 가설 전체
{json.dumps(hypotheses, ensure_ascii=False, indent=2)}

## 비교 쌍
{chr(10).join(pair_prompt_parts)}

## 평가 기준
각 쌍에 대해: 참신성, 실현 가능성, 검증 가능성, 영향력을 종합 고려하여
어느 쪽이 우수한지 판정하세요.

반드시 아래 JSON 형식으로만 답변 (마크다운 없이):
{{
  "comparisons": [
    {{"pair": [1, 2], "winner": 1, "reason": "선택 이유 (1문장)"}},
    {{"pair": [1, 3], "winner": 3, "reason": "선택 이유 (1문장)"}},
    {{"pair": [2, 3], "winner": 2, "reason": "선택 이유 (1문장)"}}
  ],
  "final_ranking": [3, 1, 2],
  "ranking_rationale": "최종 순위 근거 요약"
}}"""

    client = get_openai_client()
    model = get_gemini_model()

    # GPT + Gemini 병렬 비교
    from concurrent.futures import ThreadPoolExecutor

    def _gpt_rank():
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(r.choices[0].message.content)

    def _gem_rank():
        r = model.generate_content(prompt)
        return parse_json(r.text)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_gpt = ex.submit(_gpt_rank)
        f_gem = ex.submit(_gem_rank)
        gpt_rank = f_gpt.result()
        gem_rank = f_gem.result()

    # 승리 횟수 집계 (GPT + Gemini 합산)
    win_count: dict[int, int] = {}
    all_comparisons = []
    for src, result in [("GPT", gpt_rank), ("Gemini", gem_rank)]:
        for comp in result.get("comparisons", []):
            w = comp.get("winner", 0)
            win_count[w] = win_count.get(w, 0) + 1
            all_comparisons.append({**comp, "evaluator": src})

    # 승리 횟수 기준 내림차순 정렬
    ranking = sorted(win_count.keys(), key=lambda x: win_count[x], reverse=True)

    print(f"  → Pairwise 승리 횟수: {win_count}")
    print(f"  → 최종 순위: {ranking}")

    return {
        "ranking":           ranking,
        "win_count":         win_count,
        "pairwise_results":  all_comparisons,
        "gpt_ranking":       gpt_rank.get("final_ranking", []),
        "gemini_ranking":    gem_rank.get("final_ranking", []),
    }


# ──────────────────────────────────────────────────────────
# Round 3: Gemini Mediator (합성 가설 1-2개)
# ──────────────────────────────────────────────────────────

def _gemini_mediate(context: str, evidence_pack: str,
                   claude_draft: dict, gpt_response: dict,
                   pairwise: dict | None = None) -> dict:
    """Gemini가 Claude 3개 가설 + GPT 비판 + Pairwise 순위를 보고 검증 가능한 합성 가설 도출."""
    model = get_gemini_model()

    pairwise_section = ""
    if pairwise and pairwise.get("ranking"):
        winner_id = pairwise["ranking"][0]
        pairwise_section = f"""
## Pairwise Ranking 결과 (GPT + Gemini 쌍대비교)
- 최종 순위: {pairwise.get('ranking', [])}
- 1위 가설: #{winner_id}  (승리 횟수: {pairwise.get('win_count', {}).get(winner_id, 0)}회)
- 비교 결과: {json.dumps(pairwise.get('pairwise_results', [])[:4], ensure_ascii=False)}

⚠️ Pairwise 1위 가설(#{winner_id})을 합성의 기본 출발점으로 삼으세요.
이를 무시하거나 낮은 순위 가설을 선택하려면 명시적 근거를 synthesis_rationale에 기술하세요.
"""

    prompt = f"""당신은 딥러닝 연구 전문가입니다.
Claude의 가설 3개와 GPT의 비판, Pairwise Ranking 결과를 모두 검토하여
검증 가능하고 실현 가능한 합성 가설 1-2개를 도출하세요.

{context}

{evidence_pack}
{pairwise_section}
## Claude의 가설 3개
{json.dumps(claude_draft.get('hypotheses', []), ensure_ascii=False, indent=2)}

## GPT의 비판
{json.dumps(gpt_response, ensure_ascii=False, indent=2)}

## 합성 지침
- GPT가 지적한 결함을 해소하면서도 Claude의 혁신적 요소를 보존
- 단순 절충이 아닌 시너지를 만들 것
- 반드시 "어떤 데이터로 어떻게 검증하는가"를 포함할 것

반드시 아래 JSON 형식으로만 답변하세요 (마크다운 없이):
{{
  "synthesized_hypotheses": [
    {{
      "statement": "합성 가설 (영어 1문장)",
      "statement_kr": "합성 가설 (한국어 1문장)",
      "synthesis_rationale": "어떤 요소를 어떻게 결합했는가",
      "resolved_flaws": ["GPT 지적 중 해소된 항목 1", "항목 2"],
      "preserved_innovations": ["Claude에서 보존한 혁신 1", "혁신 2"],
      "remaining_risks": ["합성 후에도 남는 위험 1", "위험 2"],
      "verification_plan": "어떤 데이터·지표로 검증하는가"
    }}
  ],
  "recommended_id": 0,
  "for_claude_finalizer": ["최종 확정자에게 전달할 제안 1", "제안 2"]
}}"""

    response = model.generate_content(prompt)
    return parse_json(response.text)


# ──────────────────────────────────────────────────────────
# Round 4: Claude Finalizer (최종 + 반증 조건)
# ──────────────────────────────────────────────────────────

def _claude_finalize(context: str, evidence_pack: str,
                    claude_draft: dict, gpt_response: dict,
                    gemini_synthesis: dict,
                    pairwise: dict | None = None) -> dict:
    """4라운드 + Pairwise 토론을 반영한 최종 가설 + 실험 계획 + 반증 조건 확정."""
    syn = gemini_synthesis.get("synthesized_hypotheses", [{}])[0]

    pairwise_section = ""
    if pairwise and pairwise.get("ranking"):
        winner_id = pairwise["ranking"][0]
        pairwise_section = f"""
### Round 2.5 — Pairwise Ranking (GPT + Gemini 쌍대비교)
- 최종 순위: {pairwise.get('ranking', [])}  (승리 횟수: {pairwise.get('win_count', {})})
- 쌍대비교 1위: #{winner_id}
- GPT 순위: {pairwise.get('gpt_ranking', [])}  /  Gemini 순위: {pairwise.get('gemini_ranking', [])}
⚠️ Pairwise 1위 가설(#{winner_id})을 최종 확정의 기본 후보로 삼으세요.
   이를 뒤집으려면 최종 가설의 rationale에 반드시 명시적 근거를 기술하세요.
"""

    prompt = f"""당신은 딥러닝 연구 전문가입니다.
4라운드 AI 토론(Claude 초안 → GPT 비판 → Pairwise Ranking → Gemini 합성)을 거쳤습니다.
모든 토론을 반영하여 최종 연구 가설과 완전한 실험 계획을 확정하세요.

{context}

{evidence_pack}

## 토론 요약

### Round 1 — Claude 제안 가설 3개
연구 갭: {claude_draft.get('research_gap_summary', '')}
가설들: {json.dumps([h.get('statement_kr','') for h in claude_draft.get('hypotheses',[])], ensure_ascii=False)}

### Round 2 — GPT 비판
공통 약점: {gpt_response.get('critique_summary', '')}
놓친 관점: {gpt_response.get('missed_opportunities', [])}
가장 유망한 가설: #{gpt_response.get('best_hypothesis_id',1)} — {gpt_response.get('best_hypothesis_reason','')}
{pairwise_section}
### Round 3 — Gemini 합성
합성 가설: {syn.get('statement_kr', '')}
합성 근거: {syn.get('synthesis_rationale', '')}
해소된 결함: {syn.get('resolved_flaws', [])}
남은 위험: {syn.get('remaining_risks', [])}
검증 계획: {syn.get('verification_plan', '')}
Gemini 제안: {gemini_synthesis.get('for_claude_finalizer', [])}

## 임무
1. 3라운드 모든 약점이 해소되었는지 확인하고 최종 가설을 확정한다
2. 구체적인 실험 계획을 수립한다
3. **반증 조건(Falsification Criteria)**을 명시한다
   — 어떤 실험 결과가 나오면 이 가설이 틀린 것으로 판정하는가

⚠️ **중요**: 토론 결과 어떤 가설도 근거가 충분하지 않거나,
모든 가설이 치명적 결함을 갖고 있다면, 무리하게 최종 가설을 확정하지 마세요.
대신 아래 형식으로 답변하세요:
{{"status": "no_robust_hypothesis", "reason": "구체적 사유",
  "requires_human_narrowing": true,
  "suggestions": ["사용자에게 제안할 연구 방향 1", "방향 2"]}}

## 출력 형식 (JSON만)
{_HYPOTHESIS_SCHEMA}"""

    return parse_json(query_claude(prompt))


# ──────────────────────────────────────────────────────────
# 협업 모드 메인
# ──────────────────────────────────────────────────────────

def collaborative_generate(topic_file: str, papers_file: str) -> dict:
    """
    5라운드 토론 기반 가설 생성.
    Round 0: Gemini  — Evidence Pack 구조화
    Round 1: Claude  — 가설 3개 제안
    Round 2: GPT     — 공격적 비판
    Round 3: Gemini  — 합성 가설 도출
    Round 4: Claude  — 최종 확정 + 반증 조건
    """
    topic_data, papers_data, topic_name, topic_slug = _load_inputs(topic_file, papers_file)
    papers       = papers_data.get("papers", [])[:15]
    input_info   = topic_data.get("input", {})
    research_q   = topic_data.get("research_question", "")
    success_crit = topic_data.get("success_criteria", {})
    context      = _build_context(input_info, research_q, success_crit)

    debate_log = []

    # ── B-6: Coverage Check Gate ──────────────────────────
    gate_result = _coverage_check_gate(papers_data, topic_data)
    print(f"\n  [Coverage Gate] pass={gate_result['pass']}, "
          f"recommendation={gate_result.get('recommendation', '')}")
    if gate_result.get("missing_critical"):
        print(f"  [Coverage Gate] critical missing: {gate_result['missing_critical']}")
    if not gate_result["pass"]:
        recommendation = gate_result.get("recommendation", "fail_fast")
        print(f"  ⚠ [Coverage Gate] 근거 부족 → {recommendation}")
        print(f"  사유: {gate_result.get('reason', '')}")
        if gate_result.get("suggested_queries"):
            print(f"  추가 검색 제안: {gate_result['suggested_queries']}")

        result = {
            "timestamp":          datetime.now().isoformat(),
            "topic":              topic_name,
            "status":             "insufficient_evidence",
            "reason":             gate_result.get("reason", "evidence coverage 부족"),
            "evidence_coverage":  gate_result.get("evidence_coverage", {}),
            "missing_critical":   gate_result.get("missing_critical", []),
            "recommendation":     recommendation,
            "suggested_queries":  gate_result.get("suggested_queries", []),
            "generation_mode":    "collaborative",
            "debate_log":         debate_log,
        }
        from lab.config import reports_dir as _rdir
        output_path = _rdir(topic_slug) / "hypothesis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # ── Round 0: Evidence Pack ────────────────────────────
    evidence_pack, pack_list = _build_evidence_pack(papers)

    # ── Round 1: Claude 가설 3개 제안 ────────────────────
    print("\n  [Round 1 / Claude] 가설 3개 제안 중 (thinking 활성화)...")
    claude_draft = _claude_propose(context, evidence_pack)
    debate_log.append({"round": 1, "role": "Proposer", "model": "Claude",
                        "output": claude_draft})

    # insufficient_evidence 체크
    if claude_draft.get("status") == "insufficient_evidence":
        print(f"\n  ⚠ [Round 1] Evidence 부족으로 가설 생성 불가")
        print(f"  사유: {claude_draft.get('reason', '')}")
        print(f"  추가 검색 제안: {claude_draft.get('suggestions', [])}")
        result = {
            "timestamp":       datetime.now().isoformat(),
            "topic":           topic_name,
            "status":          "insufficient_evidence",
            "reason":          claude_draft.get("reason", ""),
            "suggestions":     claude_draft.get("suggestions", []),
            "generation_mode": "collaborative",
            "debate_log":      debate_log,
        }
        from lab.config import reports_dir as _rdir; output_path = _rdir(topic_slug) / "hypothesis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    hyps = claude_draft.get("hypotheses", [])
    for h in hyps:
        _print_round(1, f"Claude (Proposer) — 가설 {h.get('id')}", h.get("statement_kr", ""))

    # ── Round 2: GPT 비판 ─────────────────────────────────
    print(f"\n  [Round 2 / GPT ({OPENAI_MODEL})] 3개 가설 비판 중...")
    gpt_response = _gpt_critique(context, evidence_pack, claude_draft)
    debate_log.append({"round": 2, "role": "Critic", "model": OPENAI_MODEL,
                        "output": gpt_response})
    _print_round(2, f"GPT ({OPENAI_MODEL}) Critic",
                 f"최유망 가설: #{gpt_response.get('best_hypothesis_id')}",
                 critique=gpt_response.get("critique_summary", ""))

    # ── Round 2.5: Pairwise Ranking ──────────────────────
    print(f"\n  [Round 2.5 / GPT + Gemini] Pairwise 비교 중...")
    pairwise = _pairwise_rank(context, hyps)
    debate_log.append({"round": 2.5, "role": "PairwiseRanker", "model": "GPT+Gemini",
                        "output": pairwise})

    # ── Round 3: Gemini 합성 ──────────────────────────────
    print(f"\n  [Round 3 / Gemini ({GEMINI_MODEL})] 합성 가설 도출 중...")
    gemini_synthesis = _gemini_mediate(context, evidence_pack, claude_draft, gpt_response, pairwise)
    debate_log.append({"round": 3, "role": "Mediator", "model": GEMINI_MODEL,
                        "output": gemini_synthesis})
    syns = gemini_synthesis.get("synthesized_hypotheses", [{}])
    _print_round(3, f"Gemini ({GEMINI_MODEL}) Mediator",
                 syns[0].get("statement_kr", "") if syns else "",
                 note=syns[0].get("synthesis_rationale", "") if syns else "")

    # ── Round 4: Claude 최종 확정 ────────────────────────
    print("\n  [Round 4 / Claude] 최종 가설 확정 중 (thinking 활성화)...")
    final = _claude_finalize(context, evidence_pack, claude_draft, gpt_response, gemini_synthesis, pairwise)
    debate_log.append({"round": 4, "role": "Finalizer", "model": "Claude",
                        "output": final})

    # no_robust_hypothesis 체크
    if final.get("status") == "no_robust_hypothesis":
        print(f"\n  ⚠ [Round 4] 토론 결과 신뢰할 수 있는 가설 없음")
        print(f"  사유: {final.get('reason', '')}")
        print(f"  제안: {final.get('suggestions', [])}")
        result = {
            "timestamp":                datetime.now().isoformat(),
            "topic":                    topic_name,
            "status":                   "no_robust_hypothesis",
            "requires_human_narrowing": True,
            "reason":                   final.get("reason", ""),
            "suggestions":              final.get("suggestions", []),
            "generation_mode":          "collaborative",
            "evidence_pack":            pack_list,
            "pairwise_ranking":         pairwise,
            "debate_log":               debate_log,
        }
        from lab.config import reports_dir as _rdir; output_path = _rdir(topic_slug) / "hypothesis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    _print_round(4, "Claude (Finalizer — Final)", final.get("statement_kr", ""))

    # ── 결과 조립 ────────────────────────────────────────
    result = {
        "timestamp":        datetime.now().isoformat(),
        "topic":            topic_name,
        "papers_analyzed":  len(papers),
        "generation_mode":  "collaborative",
        "evidence_coverage": gate_result.get("evidence_coverage", {}),  # B-5
        "evidence_pack":    pack_list,
        "research_gap": {
            "summary":             final.get("research_gap_summary", ""),
            "existing_approaches": claude_draft.get("existing_approaches", []),
            "limitations":         claude_draft.get("limitations", []),
        },
        "hypothesis": {
            "statement":             final.get("statement", ""),
            "statement_kr":          final.get("statement_kr", ""),
            "rationale":             final.get("rationale", ""),
            "key_innovation":        final.get("key_innovation", ""),
            "expected_mechanism":    final.get("expected_mechanism", ""),
            "falsification_criteria": final.get("falsification_criteria", []),
        },
        "experiment_plan": {
            "architecture":       final.get("architecture", ""),
            "dataset":            final.get("dataset", ""),
            "baseline_models":    final.get("baseline_models", []),
            "evaluation_metrics": final.get("evaluation_metrics", []),
            "key_experiments":    final.get("key_experiments", []),
            "ablation_studies":   final.get("ablation_studies", []),
        },
        "related_papers": final.get("related_papers", []),
        "evidence_links": final.get("evidence_links", []),
        "confidence":     final.get("confidence", 0.0),
        "risk_factors":   final.get("risk_factors", []),
        "debate_log":     debate_log,
        "pairwise_ranking": pairwise,
        "debate_summary": {
            "claude_hypotheses":  [h.get("statement_kr","") for h in hyps],
            "gpt_critique":       gpt_response.get("critique_summary", ""),
            "gpt_best_id":        gpt_response.get("best_hypothesis_id", 1),
            "pairwise_winner":    pairwise.get("ranking", [None])[0],
            "pairwise_win_count": pairwise.get("win_count", {}),
            "gemini_synthesis":   syns[0].get("statement_kr", "") if syns else "",
            "final":              final.get("statement_kr", ""),
        },
    }

    from lab.config import reports_dir as _rdir; output_path = _rdir(topic_slug) / "hypothesis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  협업 가설 저장: {output_path}")
    return result


# ──────────────────────────────────────────────────────────
# 단일 모드: Claude 단독
# ──────────────────────────────────────────────────────────

def generate_hypothesis(topic_file: str, papers_file: str) -> dict:
    """Claude 단독으로 가설을 생성한다."""
    topic_data, papers_data, topic_name, topic_slug = _load_inputs(topic_file, papers_file)

    # B-6: Coverage Check Gate (단일 모드에서도 적용)
    gate_result = _coverage_check_gate(papers_data, topic_data)
    print(f"\n  [Coverage Gate] pass={gate_result['pass']}")
    if not gate_result["pass"]:
        print(f"  ⚠ [Coverage Gate] 근거 부족 → {gate_result.get('recommendation', 'fail_fast')}")
        result = {
            "timestamp":          datetime.now().isoformat(),
            "topic":              topic_name,
            "status":             "insufficient_evidence",
            "reason":             gate_result.get("reason", "evidence coverage 부족"),
            "evidence_coverage":  gate_result.get("evidence_coverage", {}),
            "missing_critical":   gate_result.get("missing_critical", []),
            "recommendation":     gate_result.get("recommendation", "fail_fast"),
            "suggested_queries":  gate_result.get("suggested_queries", []),
            "generation_mode":    "single",
        }
        from lab.config import reports_dir as _rdir
        output_path = _rdir(topic_slug) / "hypothesis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    papers       = papers_data.get("papers", [])[:15]
    input_info   = topic_data.get("input", {})
    research_q   = topic_data.get("research_question", "")
    success_crit = topic_data.get("success_criteria", {})
    context      = _build_context(input_info, research_q, success_crit)
    papers_text  = "\n".join([
        f"[{i+1}] {p['title']} ({p['year']})\n    요약: {p['abstract'][:300]}"
        for i, p in enumerate(papers)
    ])

    prompt = f"""당신은 딥러닝 연구 전문가입니다.
아래 연구 맥락과 논문을 바탕으로 새로운 연구 가설을 수립해주세요.

{context}

## 관련 논문
{papers_text}

## 출력 형식 (JSON만)
{_HYPOTHESIS_SCHEMA}"""

    print("  [Claude SDK] 가설 생성 중...")
    result = parse_json(query_claude(prompt))
    result.update({"timestamp": datetime.now().isoformat(),
                   "topic": topic_name, "papers_analyzed": len(papers),
                   "generation_mode": "single"})

    from lab.config import reports_dir as _rdir; output_path = _rdir(topic_slug) / "hypothesis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  가설 저장: {output_path}")
    return result


# ──────────────────────────────────────────────────────────
# 출력 & CLI
# ──────────────────────────────────────────────────────────

def print_hypothesis(result: dict) -> None:
    mode = result.get("generation_mode", "single")
    print(f"\n{'='*60}")
    print(f"  가설 생성 결과  [{mode.upper()} MODE]")
    print(f"{'='*60}")

    if mode == "collaborative":
        ds = result.get("debate_summary", {})
        print(f"\n[토론 요약]")
        for i, h in enumerate(ds.get("claude_hypotheses", [])):
            print(f"  Claude 가설 {i+1}: {h[:80]}")
        print(f"  GPT 비판:     {ds.get('gpt_critique', '')[:80]}")
        print(f"  Gemini 합성:  {ds.get('gemini_synthesis', '')[:80]}")
        print(f"  최종 가설:    {ds.get('final', '')[:80]}")
        print(f"\n{'─'*60}")

    hyp = result.get("hypothesis", {})
    print(f"\n[최종 가설]")
    print(f"  (EN) {hyp.get('statement', '')}")
    print(f"  (KR) {hyp.get('statement_kr', '')}")
    print(f"\n  근거:     {hyp.get('rationale', '')[:200]}")
    print(f"  핵심 혁신: {hyp.get('key_innovation', '')}")
    if hyp.get("falsification_criteria"):
        print(f"\n  [반증 조건]")
        for fc in hyp.get("falsification_criteria", []):
            print(f"    - {fc}")
    print(f"\n  신뢰도: {result.get('confidence', 0)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="연구 가설 생성")
    parser.add_argument("--topic-file",  required=True)
    parser.add_argument("--papers-file", required=True)
    parser.add_argument("--mode", choices=["single", "collaborative"],
                        default="single", help="생성 모드 (기본: single)")
    args = parser.parse_args()

    if args.mode == "collaborative":
        result = collaborative_generate(args.topic_file, args.papers_file)
    else:
        result = generate_hypothesis(args.topic_file, args.papers_file)

    print_hypothesis(result)
