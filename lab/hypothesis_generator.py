"""
Stage 3: 가설 생성

[단일 모드] Claude 단독 생성
  python -m lab.hypothesis_generator \
    --topic-file reports/topic_analysis.json \
    --papers-file reports/papers_{topic}.json

[협업 모드] 5라운드 토론 기반 생성 (권장)
  python -m lab.hypothesis_generator \
    --topic-file reports/topic_analysis.json \
    --papers-file reports/papers_{topic}.json \
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
  "related_papers": [
    {"title": "논문 제목", "venue": "학회", "relevance": "관련성"}
  ],
  "confidence": 0.0,
  "risk_factors": ["위험 1", "위험 2"]
}"""


def _load_inputs(topic_file: str, papers_file: str):
    topic_data  = json.loads(Path(topic_file).read_text(encoding="utf-8"))
    papers_data = json.loads(Path(papers_file).read_text(encoding="utf-8"))
    topic_name  = topic_data.get("input", {}).get("topic", "research")
    topic_slug  = re.sub(r"\W+", "_", topic_name.lower())[:30]
    return topic_data, papers_data, topic_name, topic_slug


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
# Round 0: Evidence Pack 구조화 (Gemini)
# ──────────────────────────────────────────────────────────

def _build_evidence_pack(papers: list[dict]) -> str:
    """
    각 논문을 '문제/방법/한계/실험조건/남는 빈틈' 포맷으로 구조화.
    모든 라운드가 동일한 입력을 받아 '해석 차이'가 아닌 '논리 차이'로 토론하게 한다.
    """
    model = get_gemini_model()

    papers_raw = json.dumps([
        {"id": i+1, "title": p.get("title",""), "year": p.get("year",""),
         "abstract": p.get("abstract","")[:500]}
        for i, p in enumerate(papers[:15])
    ], ensure_ascii=False, indent=2)

    prompt = f"""당신은 딥러닝 분야 전문 연구 분석가입니다.
아래 논문들을 각각 동일한 구조적 포맷으로 분석하여 Evidence Pack을 작성하세요.
모든 논문이 같은 틀로 분석되어야 나중에 연구 갭과 가설을 비교할 수 있습니다.

## 논문 목록
{papers_raw}

## 출력 형식 (JSON 배열)
반드시 아래 JSON 배열로만 답변하세요 (마크다운 없이):
[
  {{
    "id": 1,
    "title": "논문 제목",
    "problem": "이 논문이 해결하려는 핵심 문제 (1-2문장)",
    "method": "핵심 방법론 (50자 이내, 구체적 기술명 포함)",
    "limitations": "이 방법론의 핵심 한계 (1-2문장)",
    "experimental_conditions": "데이터셋 / 주요 지표 / 베이스라인",
    "remaining_gaps": "이 논문 이후에도 해결 못 한 핵심 미해결 문제"
  }},
  ...
]"""

    print("  [Round 0 / Gemini] Evidence Pack 구조화 중...")
    response = model.generate_content(prompt)
    try:
        pack_list = parse_json(response.text)
        lines = ["## Evidence Pack — 논문 구조화 분석\n"]
        for p in pack_list:
            lines.append(f"[논문 {p.get('id','')}] {p.get('title','')}")
            lines.append(f"  ▶ 문제        : {p.get('problem','')}")
            lines.append(f"  ▶ 방법        : {p.get('method','')}")
            lines.append(f"  ▶ 한계        : {p.get('limitations','')}")
            lines.append(f"  ▶ 실험 조건   : {p.get('experimental_conditions','')}")
            lines.append(f"  ▶ 남는 빈틈   : {p.get('remaining_gaps','')}")
            lines.append("")
        pack_str = "\n".join(lines)
        print(f"  → Evidence Pack 완성: {len(pack_list)}편 구조화")
        return pack_str, pack_list
    except Exception as e:
        print(f"  [경고] Evidence Pack JSON 파싱 실패: {e}")
        return text, []


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
# Round 3: Gemini Mediator (합성 가설 1-2개)
# ──────────────────────────────────────────────────────────

def _gemini_mediate(context: str, evidence_pack: str,
                   claude_draft: dict, gpt_response: dict) -> dict:
    """Gemini가 Claude 3개 가설 + GPT 비판을 보고 검증 가능한 합성 가설 도출."""
    model = get_gemini_model()

    prompt = f"""당신은 딥러닝 연구 전문가입니다.
Claude의 가설 3개와 GPT의 비판을 모두 검토하여
검증 가능하고 실현 가능한 합성 가설 1-2개를 도출하세요.

{context}

{evidence_pack}

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
                    gemini_synthesis: dict) -> dict:
    """4라운드 토론을 반영한 최종 가설 + 실험 계획 + 반증 조건 확정."""
    syn = gemini_synthesis.get("synthesized_hypotheses", [{}])[0]

    prompt = f"""당신은 딥러닝 연구 전문가입니다.
4라운드 AI 토론(Claude 초안 → GPT 비판 → Gemini 합성)을 거쳤습니다.
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

    # ── Round 0: Evidence Pack ────────────────────────────
    evidence_pack, pack_list = _build_evidence_pack(papers)

    # ── Round 1: Claude 가설 3개 제안 ────────────────────
    print("\n  [Round 1 / Claude] 가설 3개 제안 중 (thinking 활성화)...")
    claude_draft = _claude_propose(context, evidence_pack)
    debate_log.append({"round": 1, "role": "Proposer", "model": "Claude",
                        "output": claude_draft})
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

    # ── Round 3: Gemini 합성 ──────────────────────────────
    print(f"\n  [Round 3 / Gemini ({GEMINI_MODEL})] 합성 가설 도출 중...")
    gemini_synthesis = _gemini_mediate(context, evidence_pack, claude_draft, gpt_response)
    debate_log.append({"round": 3, "role": "Mediator", "model": GEMINI_MODEL,
                        "output": gemini_synthesis})
    syns = gemini_synthesis.get("synthesized_hypotheses", [{}])
    _print_round(3, f"Gemini ({GEMINI_MODEL}) Mediator",
                 syns[0].get("statement_kr", "") if syns else "",
                 note=syns[0].get("synthesis_rationale", "") if syns else "")

    # ── Round 4: Claude 최종 확정 ────────────────────────
    print("\n  [Round 4 / Claude] 최종 가설 확정 중 (thinking 활성화)...")
    final = _claude_finalize(context, evidence_pack, claude_draft, gpt_response, gemini_synthesis)
    debate_log.append({"round": 4, "role": "Finalizer", "model": "Claude",
                        "output": final})
    _print_round(4, "Claude (Finalizer — Final)", final.get("statement_kr", ""))

    # ── 결과 조립 ────────────────────────────────────────
    result = {
        "timestamp":        datetime.now().isoformat(),
        "topic":            topic_name,
        "papers_analyzed":  len(papers),
        "generation_mode":  "collaborative",
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
        },
        "related_papers": final.get("related_papers", []),
        "confidence":     final.get("confidence", 0.0),
        "risk_factors":   final.get("risk_factors", []),
        "debate_log":     debate_log,
        "debate_summary": {
            "claude_hypotheses":  [h.get("statement_kr","") for h in hyps],
            "gpt_critique":       gpt_response.get("critique_summary", ""),
            "gpt_best_id":        gpt_response.get("best_hypothesis_id", 1),
            "gemini_synthesis":   syns[0].get("statement_kr", "") if syns else "",
            "final":              final.get("statement_kr", ""),
        },
    }

    output_path = Path("reports") / topic_slug / "hypothesis.json"
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

    output_path = Path("reports") / topic_slug / "hypothesis.json"
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
