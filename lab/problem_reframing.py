"""
Problem Reframing Layer

문제를 여러 frame으로 다시 써보고 가장 실험적으로 생산적인 frame을 선택한다.
- original: 사용자 입력 그대로
- narrower_more_falsifiable: 범위를 좁혀 반증 가능성 극대화
- constraint_driven: 제약 조건 중심으로 재정의
- competing_explanation_driven: 경쟁 설명 중심으로 재정의

산출물: experiments/{slug}/reports/problem_reframing.json

사용법:
    from lab.problem_reframing import generate_reframing
    result = generate_reframing(topic_analysis, decisive_evidence)
"""

import json

from lab.config import (
    query_claude, parse_json, get_openai_client, get_gemini_model,
    OPENAI_MODEL, llm_retry, prompt_hash,
)


def _deterministic_frames(inp: dict, constraints_structured: dict) -> list[dict]:
    """규칙 기반 frame 초안을 생성한다 (LLM 없이)."""
    topic = inp.get("topic", "")
    problem = inp.get("problem_definition", "")
    constraints = inp.get("constraints", "")
    target = inp.get("target_metric", "")
    outcome = inp.get("desired_outcome", "")

    frames = []

    # Frame 0: original
    frames.append({
        "frame_id": "frame_0",
        "label": "original",
        "statement": problem or f"Solve {topic} to achieve {outcome}",
        "novelty_consequence": "As defined by user — novelty depends on literature gap",
        "falsifiability_consequence": f"Falsifiable if {target} does not meet target",
        "feasibility_consequence": f"Constrained by: {constraints}" if constraints else "No explicit constraints",
        "strongest_competing_explanation": "Unknown — requires literature review",
    })

    # Frame 1: narrower_more_falsifiable
    narrow_statement = problem
    if target:
        metrics = [m.strip() for m in target.split(",")]
        primary = metrics[0] if metrics else target
        narrow_statement = (
            f"Under {constraints if constraints else 'standard conditions'}, "
            f"can {primary} exceed the current SOTA for {topic}?"
        )
    frames.append({
        "frame_id": "frame_1",
        "label": "narrower_more_falsifiable",
        "statement": narrow_statement,
        "novelty_consequence": "Narrower scope reduces novelty breadth but sharpens contribution claim",
        "falsifiability_consequence": f"Directly falsifiable: {primary if target else 'primary metric'} vs SOTA comparison",
        "feasibility_consequence": "Easier to execute — single metric focus",
        "strongest_competing_explanation": f"SOTA method on {topic} under same constraints",
    })

    # Frame 2: constraint_driven
    cs = constraints_structured or {}
    constraint_parts = []
    if cs.get("param_budget_M"):
        constraint_parts.append(f"≤{cs['param_budget_M']}M params")
    if cs.get("single_gpu"):
        constraint_parts.append("single GPU")
    if cs.get("latency_sensitive"):
        constraint_parts.append("real-time inference")
    if cs.get("no_pretrained"):
        constraint_parts.append("no pretrained weights")
    constraint_str = ", ".join(constraint_parts) if constraint_parts else constraints

    frames.append({
        "frame_id": "frame_2",
        "label": "constraint_driven",
        "statement": (
            f"What is the best achievable {target or 'performance'} for {topic} "
            f"under strict constraints ({constraint_str})?"
        ),
        "novelty_consequence": "Novelty shifts from method innovation to constrained-regime optimization",
        "falsifiability_consequence": "Falsifiable if constraint-aware baseline matches or exceeds proposed method",
        "feasibility_consequence": "Feasibility is the primary axis — constraints define the solution space",
        "strongest_competing_explanation": "Lightweight/efficient variants of SOTA (MobileNet, EfficientNet, etc.)",
    })

    return frames


def _deterministic_frame_3(inp: dict, decisive_evidence: dict | None) -> dict:
    """4번째 frame (competing_explanation_driven)의 deterministic fallback."""
    topic = inp.get("topic", "")
    contra = []
    if decisive_evidence:
        contra = decisive_evidence.get("contra", [])

    if contra and contra[0].get("paper_id"):
        top_contra = contra[0]
        competing = top_contra.get("title", "existing method")
        competing_dim = top_contra.get("threatens_dimension", "novelty")
        statement = (
            f"Given that {competing} already addresses {competing_dim} for {topic}, "
            f"what specific gap remains that justifies a new approach?"
        )
        strongest = f"{competing} — {top_contra.get('reason', '')}"
    else:
        statement = (
            f"Assuming the strongest competing method already solves {topic} adequately, "
            f"what additional value does the proposed approach provide?"
        )
        strongest = "Best existing method under same conditions (unknown — requires literature review)"

    return {
        "frame_id": "frame_3",
        "label": "competing_explanation_driven",
        "statement": statement,
        "novelty_consequence": "Novelty must be defined RELATIVE to the strongest competitor, not in absolute terms",
        "falsifiability_consequence": "Falsifiable if proposed method does not outperform competitor on its weakest axis",
        "feasibility_consequence": "Feasibility depends on whether the proposed advantage justifies implementation cost",
        "strongest_competing_explanation": strongest,
    }


def generate_reframing(
    topic_analysis: dict,
    decisive_evidence: dict | None = None,
) -> dict:
    """문제를 여러 frame으로 재정의하고 최적 frame을 선택한다.

    Args:
        topic_analysis: topic_analysis.json
        decisive_evidence: decisive_evidence.json (없으면 pre-evidence 모드)

    Returns: problem_reframing dict
    """
    inp = topic_analysis.get("input", {})
    cs = topic_analysis.get("constraints_structured", {})

    # 1단계: 규칙 기반 frame 초안
    frames = _deterministic_frames(inp, cs)

    # 2단계: 3자 협업 — GPT(frame 제안) → Gemini(독립 평가) → Claude(최종 선택)
    contra_text = ""
    de_raw = None
    if decisive_evidence:
        de_raw = decisive_evidence.get("decisive_evidence", decisive_evidence)
        contra = de_raw.get("contra", []) if isinstance(de_raw, dict) else []
        if contra and contra[0].get("paper_id"):
            contra_text = f"Strongest contra: {contra[0].get('title', '')} — {contra[0].get('reason', '')}"

    context_block = f"""## 연구 주제
- topic: {inp.get('topic', '')}
- problem: {inp.get('problem_definition', '')}
- constraints: {inp.get('constraints', '')}
- target: {inp.get('target_metric', '')}
{f'- contra evidence: {contra_text}' if contra_text else ''}

## 기존 3개 Frame
Frame 0 (original): {frames[0]['statement']}
Frame 1 (narrower): {frames[1]['statement']}
Frame 2 (constraint): {frames[2]['statement']}"""

    try:
        # ── Step A: GPT — 4번째 frame 제안 ──
        gpt_prompt = f"""You are a research strategist. Given 3 existing problem frames,
propose a 4th frame driven by the strongest competing explanation.
This frame should ask: "What if the competitor already solved this?"

{context_block}

Return JSON only:
{{
  "frame_3": {{
    "frame_id": "frame_3",
    "label": "competing_explanation_driven",
    "statement": "problem restatement assuming competitor's approach is valid",
    "novelty_consequence": "how novelty changes under this frame",
    "falsifiability_consequence": "how falsifiability changes",
    "feasibility_consequence": "how feasibility changes",
    "strongest_competing_explanation": "the specific competitor/method this frame challenges"
  }}
}}"""
        print("    [Reframe / GPT] 4번째 frame 제안...")
        client = get_openai_client()
        gpt_resp = llm_retry(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Research strategist. Return valid JSON only."},
                {"role": "user", "content": gpt_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            label="GPT reframe",
        )
        gpt_frame = json.loads(gpt_resp.choices[0].message.content).get("frame_3", {})
        if gpt_frame:
            frames.append(gpt_frame)
            print(f"    [Reframe / GPT] frame_3: {gpt_frame.get('statement', '')[:80]}")

        # ── Step B: Gemini — 4개 frame 독립 평가 (4차원 점수 + 탈락 이유) ──
        # full frame bundle을 비교하게 함 (statement뿐 아니라 consequence 전체)
        frame_bundles = "\n\n".join(
            f"### {f['frame_id']} ({f['label']})\n"
            f"Statement: {f['statement']}\n"
            f"Novelty: {f.get('novelty_consequence','')}\n"
            f"Falsifiability: {f.get('falsifiability_consequence','')}\n"
            f"Feasibility: {f.get('feasibility_consequence','')}\n"
            f"Competing: {f.get('strongest_competing_explanation','')}"
            for f in frames
        )
        gemini_prompt = f"""You are an independent research evaluator.
Score each frame on 4 dimensions (0-1 scale) and recommend the most experimentally productive one.

Scoring weights: falsifiability 0.4, feasibility 0.3, novelty_preservation 0.2, experimental_productivity 0.1

{frame_bundles}

Return JSON only:
{{
  "recommended_frame": "frame_0|frame_1|frame_2|frame_3",
  "ranking": ["frame_1", "frame_3", "frame_2", "frame_0"],
  "frame_scores": {{
    "frame_0": {{"novelty_preservation": 0.0, "falsifiability": 0.0, "feasibility": 0.0, "experimental_productivity": 0.0}},
    "frame_1": {{"novelty_preservation": 0.0, "falsifiability": 0.0, "feasibility": 0.0, "experimental_productivity": 0.0}},
    "frame_2": {{"novelty_preservation": 0.0, "falsifiability": 0.0, "feasibility": 0.0, "experimental_productivity": 0.0}},
    "frame_3": {{"novelty_preservation": 0.0, "falsifiability": 0.0, "feasibility": 0.0, "experimental_productivity": 0.0}}
  }},
  "reason": "why recommended frame is most productive",
  "why_not_others": {{
    "frame_0": "reason for not choosing",
    "frame_2": "reason"
  }}
}}"""
        print("    [Reframe / Gemini] 4차원 평가...")
        model = get_gemini_model()
        gem_resp = llm_retry(model.generate_content, gemini_prompt, label="Gemini reframe")
        gem_text = gem_resp.text.strip()
        if "```json" in gem_text:
            gem_text = gem_text.split("```json")[1].split("```")[0].strip()
        elif "```" in gem_text:
            gem_text = gem_text.split("```")[1].split("```")[0].strip()
        gemini_eval = json.loads(gem_text)
        gemini_pick = gemini_eval.get("recommended_frame", "frame_1")
        gemini_reason = gemini_eval.get("reason", "")
        print(f"    [Reframe / Gemini] 추천: {gemini_pick}")

        # ── Step C: Claude — structured decision (Gemini 따랐는지/뒤집었는지 명시) ──
        gemini_scores = gemini_eval.get("frame_scores", {})
        gemini_why_not = gemini_eval.get("why_not_others", {})
        claude_prompt = f"""당신은 연구 전략 최종 결정자입니다.
GPT가 제안한 4번째 frame과 Gemini의 독립 평가를 참고하여 최종 working frame을 선택하세요.

{frame_bundles}

## Gemini 평가 결과
- 추천: {gemini_pick}
- 이유: {gemini_reason}
- 점수: {json.dumps(gemini_scores, ensure_ascii=False)}
- 탈락 이유: {json.dumps(gemini_why_not, ensure_ascii=False)}

## 선택 기준
- falsifiability 높고 feasibility acceptable하며 novelty 유지되는 frame
- Gemini를 따랐는지/뒤집었는지를 반드시 명시

## Override 허용 조건 (이 조건을 충족해야만 Gemini 추천을 뒤집을 수 있음)
- Gemini 1위와 2위 점수 차이가 0.1 미만
- novelty_preservation이 과도하게 희생됨
- strongest_competing_explanation 처리가 약함

반드시 아래 JSON으로만 출력:
{{
  "recommended_working_frame": "frame_0|frame_1|frame_2|frame_3",
  "followed_gemini": true,
  "override_of_gemini": false,
  "reason": "최종 선택 이유",
  "decision_basis": {{
    "falsifiability": "...",
    "feasibility": "...",
    "novelty_preservation": "...",
    "competing_explanation_handling": "..."
  }},
  "rejected_frames": [
    {{"frame_id": "frame_0", "reason": "..."}}
  ]
}}"""
        print("    [Reframe / Claude] structured decision...")
        claude_result = parse_json(query_claude(claude_prompt))
        recommended = claude_result.get("recommended_working_frame", gemini_pick)
        reason = claude_result.get("reason", gemini_reason)
        # Gemini 평가 결과를 최종 산출물에 보존
        claude_result["gemini_evaluation"] = gemini_eval
        print(f"    [Reframe / Claude] 최종: {recommended} "
              f"(followed_gemini={claude_result.get('followed_gemini', '?')})")

    except Exception as e:
        # 전체 협업 실패 시 deterministic fallback
        frame_3_fallback = _deterministic_frame_3(inp, de_raw)
        if len(frames) < 4:
            frames.append(frame_3_fallback)
        recommended = "frame_1"
        reason = f"LLM collaboration failed ({e}) — deterministic fallback to narrower_more_falsifiable"
        print(f"    [Reframe] fallback: {reason}")

    # frame 간 판단 차이 기록
    disagreements: list[str] = []
    for i, f in enumerate(frames):
        for j, g in enumerate(frames):
            if i >= j:
                continue
            if ("narrow" in f.get("novelty_consequence", "").lower()
                    and "innovat" in g.get("novelty_consequence", "").lower()):
                disagreements.append(
                    f"{f['label']} reduces novelty, {g['label']} preserves it"
                )
            if ("falsif" in f.get("falsifiability_consequence", "").lower()
                    and "falsif" not in g.get("falsifiability_consequence", "").lower()):
                disagreements.append(
                    f"{f['label']} improves falsifiability, {g['label']} does not"
                )

    result = {
        "problem_reframing": {
            "frames": frames,
            "recommended_working_frame": recommended,
            "reason": reason,
            "frame_disagreements": disagreements[:5],
        }
    }
    # Claude structured decision이 있으면 보존
    if "claude_result" in dir() and isinstance(claude_result, dict):
        result["problem_reframing"]["claude_decision"] = {
            k: claude_result.get(k)
            for k in ("followed_gemini", "override_of_gemini", "decision_basis",
                       "rejected_frames", "gemini_evaluation")
            if claude_result.get(k) is not None
        }
    return result
