"""
Stage 4: 다중 LLM 가설 검증 + 자동 개선 루프

GPT + Gemini로 가설을 검증하고, 다중 기준 미달 시 Claude가 약점을 분석하여
가설을 개선한 뒤 재검증한다.

통과 기준 (3가지 모두 충족 필요):
  1. 평균 점수 ≥ 8.5
  2. 각 항목(novelty/validity/feasibility/impact) 평균 ≥ 8.0
  3. 두 평가자 점수 차이 ≤ 1.5

Novelty 특칙: 선행연구 3~5개와 비교표 필수 → prior_art_comparison 필드

검증 전략 (critique-revise 루프):
  1. GPT + Gemini 비판적 검토 → 약점·제안·선행연구 비교표 수집
  2. Claude가 약점을 구조화된 방식으로 해소 → 가설 재작성
  3. 재검증 → 모든 기준 달성 or 최대 반복 도달 시 종료

사용법:
  # 단순 검증
  python -m lab.hypothesis_validator --hypothesis-file reports/hypothesis.json

  # 자동 개선 루프 (목표 8.5점)
  python -m lab.hypothesis_validator \\
    --hypothesis-file reports/hypothesis.json \\
    --topic-file      reports/topic_analysis.json \\
    --refine --target-score 8.5 --max-iter 3
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from lab.config import (
    OPENAI_MODEL, GEMINI_MODEL, SCORE_THRESHOLD,
    query_claude, get_openai_client, get_gemini_model, parse_json,
)


# ──────────────────────────────────────────
# 공통 검증 프롬프트
# ──────────────────────────────────────────

def _build_validation_prompt(hyp_content: dict, validation_packet: dict | None = None,
                             extra_instruction: str = "") -> str:
    """GPT/Gemini 공통 검증 프롬프트를 반환한다.

    validation_packet이 주어지면 evidence_pack, constraints, experiment_plan 등
    전체 맥락을 함께 전달하여 근거 기반 채점을 강제한다.
    """
    packet_section = ""
    scoring_caps = ""
    if validation_packet:
        vp = validation_packet
        packet_section = f"""
## 연구 맥락 (검증 패킷)

### 주제 / 문제 정의 / 제약 조건
- 주제: {vp.get('topic', '')}
- 문제 정의: {vp.get('problem_definition', '')}
- 제약 조건: {vp.get('constraints', '')}
- 성공 기준: {json.dumps(vp.get('success_criteria', {}), ensure_ascii=False)}

### Evidence Pack (구조화된 선행연구 분석)
{json.dumps(vp.get('evidence_pack', []), ensure_ascii=False, indent=2)}

### Evidence Links (가설 필드 ↔ paper_id 근거 연결)
{json.dumps(vp.get('evidence_links', []), ensure_ascii=False, indent=2)}

### Falsification Criteria (반증 조건)
{json.dumps(vp.get('falsification_criteria', []), ensure_ascii=False, indent=2)}

### 연구 갭
{json.dumps(vp.get('research_gap', {}), ensure_ascii=False, indent=2)}

### 실험 계획
{json.dumps(vp.get('experiment_plan', {}), ensure_ascii=False, indent=2)}

### 관련 논문
{json.dumps(vp.get('related_papers', []), ensure_ascii=False, indent=2)}
"""
        scoring_caps = """
## ⚠️ 점수 상한 규칙 (반드시 적용)
- evidence_pack의 paper_id를 인용하지 못하면 → novelty 최대 6점
- evidence_links가 비어 있거나 paper_id가 evidence_pack과 불일치하면 → novelty/validity 최대 6점
- constraints를 언급하지 않으면 → feasibility 최대 6점
- baseline / metric / falsification criteria가 없으면 → validity 최대 6점
- 위 규칙을 어기고 높은 점수를 주지 마세요. 근거 없는 고점은 무의미합니다.
"""

    return f"""당신은 딥러닝 연구 전문가입니다. 아래 연구 가설을 비판적으로 검토해주세요.
{packet_section}
## 연구 가설
{json.dumps(hyp_content, ensure_ascii=False, indent=2)}

## 검토 기준
1. 참신성 (Novelty): 기존 연구 대비 새로운 기여가 있는가?
   ⚠️ **Novelty 채점 필수 조건**: 반드시 evidence_pack 또는 선행연구 3~5개와 이 가설의 차이점을 표 형식으로 비교한 뒤에 점수를 부여하세요.
   비교표 형식 (prior_art_comparison 필드):
   | 선행연구 | 핵심 방법 | 본 가설과의 차이 |
   각 행마다 선행연구명, 핵심방법, 차이점을 명시하세요.

2. 타당성 (Validity): 가설이 논리적으로 일관성 있는가?
3. 실현 가능성 (Feasibility): 현재 기술로 구현 가능한가?
4. 영향력 (Impact): 성공 시 해당 분야에 의미 있는 기여를 하는가?
{scoring_caps}
각 기준을 고려하여 10점 만점으로 채점하세요 (9-10: 탁월, 7-8: 양호, 5-6: 보통, <5: 부족).
통과 기준: 평균 ≥ {SCORE_THRESHOLD}, 각 항목 ≥ 8.0.
{extra_instruction}
반드시 아래 JSON 형식으로만 답변하세요 (마크다운 없이):
{{
  "score": 점수(1-10),
  "score_breakdown": {{"novelty": 0, "validity": 0, "feasibility": 0, "impact": 0}},
  "prior_art_comparison": [
    {{"paper": "선행연구명", "method": "핵심 방법", "difference": "본 가설과의 차이"}}
  ],
  "evidence_used": ["paper_id_1", "paper_id_2"],
  "constraint_considered": true,
  "feasibility": "실현 가능성 평가",
  "strengths": ["강점1", "강점2"],
  "weaknesses": ["약점1 (구체적으로)", "약점2 (구체적으로)"],
  "suggestions": ["개선안1 (실행 가능한 수준으로)", "개선안2"],
  "verdict": "approve/revise/reject"
}}"""


# ──────────────────────────────────────────
# 다중 기준 통과 판정
# ──────────────────────────────────────────

def _passes_validation(gpt: dict, gem: dict, target_score: float = SCORE_THRESHOLD) -> bool:
    """
    세 가지 기준을 모두 충족해야 통과:
      1. 평균 점수 ≥ target_score
      2. 각 항목 최저점 ≥ 8.0  (novelty/validity/feasibility/impact 개별 평균)
      3. 평가자 간 점수 차이 ≤ 1.5
    """
    gpt_score = gpt.get("score", 0)
    gem_score = gem.get("score", 0)
    avg = (gpt_score + gem_score) / 2

    # 기준 1: 평균
    if avg < target_score:
        return False

    # 기준 2: 항목별 최저점
    criteria = ["novelty", "validity", "feasibility", "impact"]
    gpt_bd = gpt.get("score_breakdown", {})
    gem_bd = gem.get("score_breakdown", {})
    for c in criteria:
        item_avg = (gpt_bd.get(c, 0) + gem_bd.get(c, 0)) / 2
        if item_avg < 8.0:
            return False

    # 기준 3: 평가자 간 점수 차이
    if abs(gpt_score - gem_score) > 1.5:
        return False

    return True


# ──────────────────────────────────────────
# GPT-4o 검증
# ──────────────────────────────────────────

def validate_with_gpt4o(hypothesis: dict, validation_packet: dict | None = None) -> dict:
    hyp_content = hypothesis.get("hypothesis", hypothesis)
    prompt = _build_validation_prompt(hyp_content, validation_packet)
    response = get_openai_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ──────────────────────────────────────────
# Gemini 검증
# ──────────────────────────────────────────

def validate_with_gemini(hypothesis: dict, validation_packet: dict | None = None) -> dict:
    hyp_content = hypothesis.get("hypothesis", hypothesis)
    prompt = _build_validation_prompt(hyp_content, validation_packet)
    response = get_gemini_model().generate_content(prompt)
    return parse_json(response.text)


# ──────────────────────────────────────────
# Claude 가설 개선 (critique-revise)
# ──────────────────────────────────────────

def refine_hypothesis_with_claude(
    hypothesis: dict,
    validation: dict,
    topic: dict,
    iteration: int,
    lint_warnings: list[str] | None = None,
    validation_packet: dict | None = None,
) -> dict:
    """
    GPT-4o + Gemini의 피드백을 바탕으로 Claude가 가설을 개선한다.

    개선 전략:
    - 약점마다 구체적인 수정 방향을 도출
    - 기존 강점은 보존하면서 약점만 해소
    - 실현 가능성을 낮추지 않는 범위에서 참신성 강화
    """
    summary = validation.get("summary", {})
    gpt  = validation.get("gpt4o", {})
    gem  = validation.get("gemini", {})
    inp  = topic.get("input", {}) if topic else {}

    all_weaknesses  = list(dict.fromkeys(
        gpt.get("weaknesses", []) + gem.get("weaknesses", [])
    ))
    all_suggestions = list(dict.fromkeys(
        gpt.get("suggestions", []) + gem.get("suggestions", [])
    ))
    gpt_breakdown = gpt.get("score_breakdown", {})
    gem_breakdown = gem.get("score_breakdown", {})

    # 점수 낮은 기준 파악
    criteria = ["novelty", "validity", "feasibility", "impact"]
    avg_breakdown = {
        c: round((gpt_breakdown.get(c, 5) + gem_breakdown.get(c, 5)) / 2, 1)
        for c in criteria
    }
    weak_criteria = [c for c, v in avg_breakdown.items() if v < 8]

    # Evidence Pack 섹션 구성
    evidence_section = ""
    if validation_packet:
        vp = validation_packet
        evidence_section = f"""
## Evidence Pack (개선 시 반드시 인용)
{json.dumps(vp.get('evidence_pack', []), ensure_ascii=False, indent=2)}

## Evidence Links (현재 근거 연결 — 유지 또는 수정)
{json.dumps(vp.get('evidence_links', []), ensure_ascii=False, indent=2)}

## Falsification Criteria (현재 반증 조건)
{json.dumps(vp.get('falsification_criteria', []), ensure_ascii=False, indent=2)}

## 연구 갭
{json.dumps(vp.get('research_gap', {}), ensure_ascii=False, indent=2)}

## 기존 실험 계획
{json.dumps(vp.get('experiment_plan', {}), ensure_ascii=False, indent=2)}

## 관련 논문
{json.dumps(vp.get('related_papers', []), ensure_ascii=False, indent=2)}
"""

    prompt = f"""당신은 딥러닝 연구 전문가입니다.
아래 연구 가설이 LLM 검증에서 {summary.get('average_score', 0)}/10점을 받았습니다.
목표 점수 {SCORE_THRESHOLD}점에 미달하여 가설을 개선해야 합니다.
이번이 {iteration}번째 개선 시도입니다.

## 현재 연구 패키지 (전체)
{json.dumps({
    "hypothesis":            hypothesis.get("hypothesis", hypothesis),
    "research_gap":          hypothesis.get("research_gap", {}),
    "experiment_plan":       hypothesis.get("experiment_plan", {}),
    "related_papers":        hypothesis.get("related_papers", []),
    "falsification_criteria":hypothesis.get("falsification_criteria", []),
    "evidence_links":        hypothesis.get("evidence_links", []),
    "risk_factors":          hypothesis.get("risk_factors", []),
}, ensure_ascii=False, indent=2)}

## 연구 맥락
- 주제: {inp.get('topic', '')}
- 문제 정의: {inp.get('problem_definition', '')}
- 원하는 결과: {inp.get('desired_outcome', '')}
- 제약 조건: {inp.get('constraints', '')}
{evidence_section}
## 점수 분석 (평균)
{json.dumps(avg_breakdown, ensure_ascii=False)}
→ 개선 필요 기준: {', '.join(weak_criteria) if weak_criteria else '전반적 강화 필요'}

## GPT-4o 지적 약점
{json.dumps(all_weaknesses, ensure_ascii=False)}

## 개선 제안 (GPT + Gemini)
{json.dumps(all_suggestions, ensure_ascii=False)}

## 구조 검사 경고 (반드시 해결)
{json.dumps(lint_warnings or [], ensure_ascii=False)}

## 개선 지침
1. 각 약점에 대해 **구체적인 해소 방안**을 가설에 반영하라
2. 기존 강점({', '.join(gpt.get('strengths', [])[:2])})은 반드시 유지하라
3. {', '.join(weak_criteria)} 기준을 집중적으로 강화하라
4. 가설은 여전히 실험 가능한 수준을 유지하라
5. 위 구조 검사 경고가 있다면 **반드시 해당 필드를 보충하라**
6. Evidence Pack이 있다면 수정한 각 필드(rationale, key_innovation 등)에 paper_id를 인용하라
   형식 예: "... (evidence_id: arxiv:2204.04524)" — evidence_pack에 있는 paper_id만 사용할 것
7. evidence_links는 반드시 재작성하되, **모든 paper_id는 Evidence Pack의 유효 ID와 일치**해야 한다
   Evidence Pack에 없는 paper_id는 evidence_links에 절대 포함하지 말 것
8. evidence_links의 supports 값은 아래 허용 집합만 사용하라:
   rationale, key_innovation, expected_mechanism, falsification_criteria,
   architecture, dataset, baseline_models, evaluation_metrics, key_experiments, ablation_studies

## 출력 형식 (JSON만 — 전체 연구 패키지를 재작성)
{{
  "research_gap_summary": "연구 갭 요약 (개선 반영)",
  "statement": "개선된 가설 (영어 1문장)",
  "statement_kr": "개선된 가설 (한국어 1문장)",
  "rationale": "개선 근거 (기존과 무엇이 달라졌는지 명시)",
  "key_innovation": "핵심 혁신 포인트",
  "expected_mechanism": "예상 작동 원리",
  "falsification_criteria": ["반증 조건 1", "반증 조건 2"],
  "architecture": "제안 모델 구조 (개선 반영)",
  "dataset": "데이터셋 제안",
  "baseline_models": ["기준 모델 1 (최소 2개)", "기준 모델 2"],
  "evaluation_metrics": ["지표 1", "지표 2"],
  "key_experiments": ["핵심 실험 1", "핵심 실험 2"],
  "ablation_studies": ["ablation 1", "ablation 2"],
  "related_papers": [{{"paper_id": "paper-id", "title": "논문명", "venue": "학회", "relevance": "관련성"}}],
  "evidence_links": [{{"paper_id": "paper-id", "supports": ["key_innovation", "expected_mechanism"]}}],
  "risk_factors": ["위험 1", "위험 2"],
  "improvement_log": "이번 개선에서 수정한 내용 요약"
}}"""

    print(f"    [Claude SDK] 가설 개선 중 (iteration {iteration})...")
    refined = parse_json(query_claude(prompt))

    # 전체 연구 패키지를 필드 분류별로 명시적 동기화
    updated = dict(hypothesis)
    updated.setdefault("hypothesis", {})
    updated.setdefault("experiment_plan", {})

    # hypothesis 내부 필드
    _HYP_FIELDS = ("statement", "statement_kr", "rationale",
                   "key_innovation", "expected_mechanism")
    for field in _HYP_FIELDS:
        if field in refined:
            updated["hypothesis"][field] = refined[field]

    # experiment_plan 필드
    _EXP_FIELDS = ("architecture", "dataset", "baseline_models",
                   "evaluation_metrics", "key_experiments", "ablation_studies")
    for field in _EXP_FIELDS:
        if field in refined:
            updated["experiment_plan"][field] = refined[field]

    # 최상위 패키지 필드
    _PKG_FIELDS = ("related_papers", "risk_factors", "falsification_criteria", "evidence_links")
    for field in _PKG_FIELDS:
        if field in refined:
            updated[field] = refined[field]

    # falsification_criteria를 nested hypothesis에도 동기화
    if "falsification_criteria" in refined:
        updated["hypothesis"]["falsification_criteria"] = refined["falsification_criteria"]

    updated["_refinement_log"] = updated.get("_refinement_log", [])
    updated["_refinement_log"].append({
        "iteration":   iteration,
        "prev_score":  summary.get("average_score", 0),
        "improvement": refined.get("improvement_log", ""),
    })
    return updated


# ──────────────────────────────────────────
# 단순 검증
# ──────────────────────────────────────────

def _build_validation_packet(hypothesis: dict, topic: dict | None = None) -> dict | None:
    """hypothesis + topic 데이터로부터 검증 패킷을 구성한다."""
    if not topic:
        return None
    inp = topic.get("input", {})
    return {
        "topic":              inp.get("topic", ""),
        "problem_definition": inp.get("problem_definition", ""),
        "constraints":        inp.get("constraints", ""),
        "success_criteria":   topic.get("success_criteria", {}),
        "evidence_pack":      hypothesis.get("evidence_pack", []),
        "research_gap":       hypothesis.get("research_gap", {}),
        "experiment_plan":    hypothesis.get("experiment_plan", {}),
        "related_papers":     hypothesis.get("related_papers", []),
        "falsification_criteria": hypothesis.get(
            "falsification_criteria",
            hypothesis.get("hypothesis", {}).get("falsification_criteria", []),
        ),
        "evidence_links": hypothesis.get("evidence_links", []),
    }


def _lint_hypothesis(hypothesis: dict) -> list[str]:
    """LLM 채점 전 rule-based 구조 검사.

    필수 요소가 빠져 있으면 해당 항목을 warnings 리스트로 반환한다.
    warnings가 있으면 refine으로 전송하여 구조부터 보완시킨다.
    """
    warnings = []
    hyp = hypothesis.get("hypothesis", hypothesis)
    exp = hypothesis.get("experiment_plan", {})
    gap = hypothesis.get("research_gap", {})
    evidence = hypothesis.get("evidence_pack", [])
    falsif = hypothesis.get("falsification_criteria", hyp.get("falsification_criteria", []))
    related = hypothesis.get("related_papers", [])

    # 1. 핵심 5요소: intervention / comparator / metric / dataset / expected_effect
    required_fields = {
        "key_innovation":      "intervention (핵심 혁신/개입)",
        "expected_mechanism":  "expected_effect (예상 작동 원리)",
    }
    for field, label in required_fields.items():
        if not hyp.get(field):
            warnings.append(f"[구조] {label} 누락")

    if not exp.get("evaluation_metrics"):
        warnings.append("[구조] evaluation_metrics (평가 지표) 누락")
    if not exp.get("dataset") and not exp.get("datasets"):
        warnings.append("[구조] dataset 정보 누락")
    baselines = exp.get("baselines", exp.get("baseline_models", []))
    if len(baselines) < 2:
        warnings.append(f"[구조] baseline {len(baselines)}개 — 최소 2개 필요")

    # 2. Falsification criteria
    if not falsif or len(falsif) < 1:
        warnings.append("[구조] falsification_criteria 누락 — 반증 조건 최소 1개 필요")

    # 3. Ablation 계획
    ablation = exp.get("ablation_plan", exp.get("ablation_studies", []))
    if not ablation:
        warnings.append("[구조] ablation 계획 누락")

    # 4. Prior-art reference 연결
    if not related and not evidence:
        warnings.append("[근거] related_papers와 evidence_pack 모두 비어 있음")

    # 4-b. evidence_links 존재 및 형식 검사
    evidence_links = hypothesis.get("evidence_links", [])
    if evidence and not evidence_links:
        warnings.append("[근거] evidence_pack은 있으나 evidence_links 누락")
    if evidence_links and not isinstance(evidence_links, list):
        warnings.append("[근거] evidence_links 형식 오류")

    # 5. Research gap과 hypothesis의 연결
    if gap:
        gap_text = json.dumps(gap, ensure_ascii=False).lower()
        stmt = hyp.get("statement", "").lower() + hyp.get("statement_kr", "").lower()
        # gap에서 핵심 단어가 hypothesis에 반영됐는지 간단 확인
        gap_keywords = [w for w in gap_text.split() if len(w) > 4][:5]
        overlap = sum(1 for kw in gap_keywords if kw in stmt)
        if gap_keywords and overlap == 0:
            warnings.append("[연결] research_gap 키워드가 hypothesis statement에 미반영")

    return warnings


_EVIDENCE_LINK_SUPPORTS = {
    "rationale", "key_innovation", "expected_mechanism", "falsification_criteria",
    "architecture", "dataset", "baseline_models", "evaluation_metrics",
    "key_experiments", "ablation_studies",
}


def _audit_evidence_links(evidence_links: list, valid_ids: set[str]) -> list[str]:
    """evidence_links의 구조·paper_id 유효성·supports 값을 검사하고 오류 목록을 반환한다."""
    if not isinstance(evidence_links, list) or not evidence_links:
        return ["evidence_links missing"]

    errors = []
    for i, link in enumerate(evidence_links):
        if not isinstance(link, dict):
            errors.append(f"evidence_links[{i}] not dict")
            continue

        pid      = str(link.get("paper_id", "")).strip()
        supports = link.get("supports", [])

        if not pid:
            errors.append(f"evidence_links[{i}] missing paper_id")
        elif pid not in valid_ids:
            errors.append(f"evidence_links[{i}] invalid paper_id={pid}")

        if not isinstance(supports, list) or not supports:
            errors.append(f"evidence_links[{i}] missing supports")
        else:
            bad = [s for s in supports if s not in _EVIDENCE_LINK_SUPPORTS]
            if bad:
                errors.append(f"evidence_links[{i}] invalid supports={bad}")

    return errors


def _enforce_caps(result: dict, validation_packet: dict | None) -> dict:
    """LLM 응답에서 근거 미충족 시 항목별 점수 상한을 코드 수준에서 강제한다.

    프롬프트 지시만으로는 LLM이 규칙을 무시할 수 있으므로
    evidence_used / prior_art_comparison / constraint_considered 필드를
    실제 evidence_pack ID와 대조하여 직접 캡을 적용한다.
    """
    bd          = result.setdefault("score_breakdown", {})
    cap_reasons = result.setdefault("cap_reasons", [])

    evidence_pack = validation_packet.get("evidence_pack", []) if validation_packet else []
    valid_ids = {
        str(p.get("paper_id")).strip()
        for p in evidence_pack
        if p.get("paper_id")
    }

    used_ids    = [str(x).strip() for x in result.get("evidence_used", []) if str(x).strip()]
    invalid_ids = [x for x in used_ids if x not in valid_ids]
    prior_art   = result.get("prior_art_comparison", [])
    has_prior_art = isinstance(prior_art, list) and len(prior_art) >= 1

    # novelty: evidence_used 없음 / 유효하지 않은 ID / prior_art 없음 → 최대 6점
    if validation_packet:
        if not used_ids:
            bd["novelty"] = min(bd.get("novelty", 0), 6)
            cap_reasons.append("novelty capped: evidence_used missing")
        elif invalid_ids:
            bd["novelty"] = min(bd.get("novelty", 0), 6)
            cap_reasons.append(f"novelty capped: invalid evidence_used {invalid_ids}")
        elif not has_prior_art:
            bd["novelty"] = min(bd.get("novelty", 0), 6)
            cap_reasons.append("novelty capped: prior_art_comparison missing")

    # feasibility: constraint 미고려 시 최대 6점
    if validation_packet and not result.get("constraint_considered", False):
        bd["feasibility"] = min(bd.get("feasibility", 0), 6)
        cap_reasons.append("feasibility capped: constraint_considered=false")

    # validity: validation_packet 있을 때만 baseline + metric + falsification_criteria 검사
    # (topic 없는 단순 검증 시 구조 부재만으로 자동 감점되지 않도록)
    if validation_packet:
        exp    = validation_packet.get("experiment_plan", {})
        falsif = validation_packet.get("falsification_criteria", [])
        has_baseline = bool(exp.get("baseline_models") or exp.get("baselines"))
        has_metric   = bool(exp.get("evaluation_metrics"))
        if not (has_baseline and has_metric and falsif):
            bd["validity"] = min(bd.get("validity", 0), 6)
            cap_reasons.append("validity capped: baseline/metric/falsification incomplete")

    # evidence_links 감사: 구조 오류·잘못된 paper_id 있으면 novelty + validity 동시 cap
    evidence_links = validation_packet.get("evidence_links", []) if validation_packet else []
    link_errors = _audit_evidence_links(evidence_links, valid_ids)
    if validation_packet and link_errors:
        bd["novelty"]  = min(bd.get("novelty",  0), 6)
        bd["validity"] = min(bd.get("validity", 0), 6)
        cap_reasons.append(f"evidence_links audit failed: {link_errors}")

    # score를 score_breakdown 평균으로 재계산
    criteria = ("novelty", "validity", "feasibility", "impact")
    result["score"] = round(sum(bd.get(k, 0) for k in criteria) / len(criteria), 1)
    return result


def _score_and_build(hypothesis: dict, target_score: float = SCORE_THRESHOLD,
                     topic: dict | None = None) -> dict:
    """GPT + Gemini 병렬 검증 후 validation dict 반환. lint 실패 시 자동 revise."""
    # Rule-based lint
    lint_warnings = _lint_hypothesis(hypothesis)
    if lint_warnings:
        print(f"    [Lint] 구조 검사 경고 {len(lint_warnings)}건:")
        for w in lint_warnings:
            print(f"      - {w}")

    validation_packet = _build_validation_packet(hypothesis, topic)
    print(f"    [{OPENAI_MODEL} + {GEMINI_MODEL}] 병렬 검증 중...")
    if validation_packet:
        print(f"    → 검증 패킷 포함: evidence_pack {len(validation_packet.get('evidence_pack', []))}편, "
              f"constraints: {'있음' if validation_packet.get('constraints') else '없음'}")
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_gpt = executor.submit(validate_with_gpt4o, hypothesis, validation_packet)
        fut_gem = executor.submit(validate_with_gemini, hypothesis, validation_packet)
        gpt = _enforce_caps(fut_gpt.result(), validation_packet)
        gem = _enforce_caps(fut_gem.result(), validation_packet)
    print(f"    → GPT: {gpt.get('score')}/10  Gemini: {gem.get('score')}/10")

    gpt_score = gpt.get("score", 0)
    gem_score = gem.get("score", 0)
    avg = round((gpt_score + gem_score) / 2, 1)
    score_diff = round(abs(gpt_score - gem_score), 1)

    # 항목별 평균
    criteria = ["novelty", "validity", "feasibility", "impact"]
    gpt_bd = gpt.get("score_breakdown", {})
    gem_bd = gem.get("score_breakdown", {})
    avg_breakdown = {
        c: round((gpt_bd.get(c, 0) + gem_bd.get(c, 0)) / 2, 1)
        for c in criteria
    }

    passed = _passes_validation(gpt, gem, target_score)
    verdicts = [gpt.get("verdict"), gem.get("verdict")]

    # Lint 경고가 3건 이상이면 LLM 점수와 무관하게 revise 강제
    lint_forced_revise = len(lint_warnings) >= 3
    if lint_forced_revise:
        passed = False
        print(f"    ⚠ Lint 경고 {len(lint_warnings)}건 → 구조 보완 필요, 강제 revise")

    overall = (
        "approve" if passed
        else "reject" if verdicts.count("reject") >= 2
        else "revise"
    )

    print(f"    → 평균: {avg}/10  항목별: {avg_breakdown}  평가자차이: {score_diff}  통과: {passed}")

    return {
        "gpt4o":  gpt,
        "gemini": gem,
        "lint_warnings": lint_warnings,
        "summary": {
            "average_score":    avg,
            "avg_breakdown":    avg_breakdown,
            "score_diff":       score_diff,
            "passed_criteria":  passed,
            "overall_verdict":  overall,
            "lint_warning_count": len(lint_warnings),
            "all_strengths":    gpt.get("strengths", []) + gem.get("strengths", []),
            "all_weaknesses":   gpt.get("weaknesses", []) + gem.get("weaknesses", []),
            "all_suggestions":  gpt.get("suggestions", []) + gem.get("suggestions", []),
        },
    }


def validate_hypothesis(hypothesis_file: str, topic_file: str = "",
                        target_score: float = SCORE_THRESHOLD) -> dict:
    """GPT + Gemini 단순 1회 검증 (다중 기준 판정 포함)."""
    hypothesis_path = Path(hypothesis_file)
    if not hypothesis_path.exists():
        raise FileNotFoundError(f"가설 파일 없음: {hypothesis_file}")

    hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    topic_path = Path(topic_file) if topic_file else None
    topic = json.loads(topic_path.read_text(encoding="utf-8")) if (topic_path and topic_path.exists()) else None
    validation = _score_and_build(hypothesis, target_score, topic)
    validation["timestamp"] = datetime.now().isoformat()
    validation["hypothesis_file"] = hypothesis_file

    reports_dir = Path(hypothesis_file).parent
    output_path = reports_dir / "validation.json"
    output_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    결과 저장: {output_path}")
    return validation


# ──────────────────────────────────────────
# 자동 개선 루프
# ──────────────────────────────────────────

def refine_and_validate(
    hypothesis_file: str,
    topic_file: str = "",
    target_score: float = SCORE_THRESHOLD,
    max_iterations: int = 3,
) -> dict:
    """
    목표 점수에 달할 때까지 critique-revise 루프를 반복한다.

    흐름:
      검증 → 점수 확인 → (미달) Claude 개선 → 재검증 → ...
                       → (달성) 완료

    Returns:
        최종 validation dict (iteration_history 포함)
    """
    hypothesis_path = Path(hypothesis_file)
    topic_path = Path(topic_file) if topic_file else None

    hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    topic = json.loads(topic_path.read_text(encoding="utf-8")) if (topic_path and topic_path.exists()) else {}

    # reports_dir: hypothesis.json 의 부모 디렉터리 (= reports/{slug}/)
    reports_dir = hypothesis_path.parent

    history = []
    best_validation = None
    best_score = 0.0
    current_hypothesis = hypothesis

    print(f"\n  [검증 루프] 목표 점수: {target_score}/10  최대 {max_iterations}회")
    print("=" * 60)

    for i in range(1, max_iterations + 1):
        print(f"\n  ── Iteration {i}/{max_iterations} ──")
        validation = _score_and_build(current_hypothesis, target_score, topic or None)
        validation_packet = _build_validation_packet(current_hypothesis, topic or None)
        avg = validation["summary"]["average_score"]
        passed = validation["summary"]["passed_criteria"]
        avg_bd = validation["summary"]["avg_breakdown"]
        diff = validation["summary"]["score_diff"]
        print(f"  → 평균: {avg}/10  통과: {passed}  (항목최저: {min(avg_bd.values()):.1f}, 평가자차이: {diff})")

        history.append({
            "iteration": i,
            "score":     avg,
            "passed":    passed,
            "avg_breakdown": avg_bd,
            "score_diff": diff,
            "hypothesis_snapshot": current_hypothesis.get("hypothesis", {}).get("statement_kr", ""),
        })

        if avg > best_score:
            best_score = avg
            best_validation = validation
            best_validation["hypothesis"] = current_hypothesis  # 최고 점수 가설 보존

        if passed:
            print(f"\n  모든 검증 기준 통과! (평균 {avg} ≥ {target_score}, 항목최저 {min(avg_bd.values()):.1f} ≥ 8.0, 평가자차이 {diff} ≤ 1.5)")
            break

        # 미통과 이유 출력
        reasons = []
        if avg < target_score:
            reasons.append(f"평균 {avg} < {target_score}")
        low_items = [c for c, v in avg_bd.items() if v < 8.0]
        if low_items:
            reasons.append(f"항목 미달: {', '.join(low_items)}")
        if diff > 1.5:
            reasons.append(f"평가자 점수차 {diff} > 1.5")
        print(f"  미통과 이유: {' / '.join(reasons)}")

        if i < max_iterations:
            print(f"  → Claude 가설 개선 시작...")
            current_hypothesis = refine_hypothesis_with_claude(
                current_hypothesis, validation, topic, i,
                lint_warnings=validation.get("lint_warnings", []),
                validation_packet=validation_packet,
            )
            # 개선된 가설을 파일에 저장
            refined_path = reports_dir / f"hypothesis_refined_iter{i}.json"
            refined_path.write_text(
                json.dumps(current_hypothesis, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"  개선 가설 저장: {refined_path}")
        else:
            print(f"\n  최대 반복 도달 ({max_iterations}회). 최고 점수: {best_score}/10")

    # 최종 결과 저장
    goal_achieved = best_validation["summary"].get("passed_criteria", False)
    best_validation["timestamp"] = datetime.now().isoformat()
    best_validation["hypothesis_file"] = hypothesis_file
    best_validation["refinement_summary"] = {
        "iterations_run":  len(history),
        "target_score":    target_score,
        "achieved_score":  best_score,
        "goal_achieved":   goal_achieved,
        "iteration_history": history,
    }

    # 최종 validation 저장
    output_val = reports_dir / "validation.json"
    output_val.write_text(json.dumps(best_validation, ensure_ascii=False, indent=2), encoding="utf-8")

    if goal_achieved and len(history) > 1:
        # 최종 개선된 가설 덮어쓰기
        hypothesis_path.write_text(
            json.dumps(best_validation["hypothesis"], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  최종 개선 가설 저장: {hypothesis_file}")

    print(f"\n  검증 완료  |  최종 점수: {best_score}/10  |  목표 달성: {goal_achieved}")
    return best_validation


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다중 LLM 가설 검증 + 자동 개선")
    parser.add_argument("--hypothesis-file", required=True)
    parser.add_argument("--topic-file",      default="", help="개선 루프에 필요한 주제 파일")
    parser.add_argument("--refine",          action="store_true", help="자동 개선 루프 활성화")
    parser.add_argument("--target-score",    type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--max-iter",        type=int,   default=3)
    args = parser.parse_args()

    if args.refine:
        result = refine_and_validate(
            args.hypothesis_file,
            args.topic_file,
            args.target_score,
            args.max_iter,
        )
        rs = result["refinement_summary"]
        print(f"\n최종 점수: {rs['achieved_score']}/10  |  목표 달성: {rs['goal_achieved']}")
        for h in rs["iteration_history"]:
            print(f"  Iter {h['iteration']}: {h['score']}/10")
    else:
        result = validate_hypothesis(args.hypothesis_file, args.topic_file, args.target_score)
        print(f"\n종합 평점: {result['summary']['average_score']}/10")
        print(f"종합 판정: {result['summary']['overall_verdict']}")
