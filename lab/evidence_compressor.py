"""
Decisive Evidence Compressor

evidence_pack + coverage 정보에서 판세를 바꾸는 핵심 근거를 압축한다.
- support: 가설을 가장 강하게 지지하는 근거
- contra: 가설을 가장 강하게 위협하는 근거 (strongest competing explanation 포함)
- swing: 확인 여부에 따라 판정이 뒤집히는 근거
- decision_pressure: 가장 약한 차원 (novelty/validity/feasibility)

산출물: experiments/{slug}/reports/decisive_evidence.json

사용법:
    from lab.evidence_compressor import compress_decisive_evidence
    result = compress_decisive_evidence(papers, coverage, group_coverage, must_cover)
"""

import json

from lab.config import (
    COVERAGE_GROUPS, get_openai_client, get_gemini_model,
    OPENAI_MODEL, llm_retry, prompt_hash,
)

# ── importance score 가중치 ─────────────────────────────

_ROLE_WEIGHT = {
    "closest_prior_art":     0.35,
    "supporting_mechanism":  0.25,
    "baseline_reference":    0.20,
    "evaluation_reference":  0.15,
    "constraint_reference":  0.15,
    "failure_mode_reference": 0.20,
    "falsification_reference": 0.30,
}

_STRENGTH_WEIGHT = {"direct": 1.0, "indirect": 0.6, "contextual": 0.3}

# contra 역할: 가설을 위협하는 역할
_CONTRA_ROLES = {"closest_prior_art", "falsification_reference", "failure_mode_reference"}

# support 역할
_SUPPORT_ROLES = {"supporting_mechanism", "baseline_reference", "evaluation_reference",
                  "constraint_reference"}


def _importance_score(paper: dict) -> float:
    """논문의 decisive importance를 계산한다."""
    role = paper.get("evidence_role", "")
    strength = paper.get("support_strength", "contextual")
    rank = paper.get("rank", 99)
    n_slots = len(paper.get("claim_slots_supported", []))

    role_w = _ROLE_WEIGHT.get(role, 0.1)
    strength_w = _STRENGTH_WEIGHT.get(strength, 0.3)
    rank_w = max(0.1, 1.0 - rank * 0.05)  # rank 1 → 0.95, rank 20 → 0.0
    slot_w = min(1.0, n_slots * 0.15)      # 슬롯 많을수록 보너스 (최대 1.0)

    return round(role_w * 0.35 + strength_w * 0.30 + rank_w * 0.20 + slot_w * 0.15, 4)


def _dimension_for_slots(slots: list[str]) -> str:
    """claim_slots에서 가장 관련 깊은 차원을 결정한다."""
    for dim, dim_slots in COVERAGE_GROUPS.items():
        if any(s in dim_slots for s in slots):
            return dim
    return "impact"


def compress_decisive_evidence(
    papers: list[dict],
    coverage: dict[str, bool],
    group_coverage: dict[str, bool],
    must_cover: list[str],
) -> dict:
    """판세를 바꾸는 핵심 근거를 support/contra/swing으로 압축한다.

    Args:
        papers: papers.json의 논문 리스트 (evidence_role, support_strength 포함)
        coverage: slot별 커버 여부 {slot: bool}
        group_coverage: 그룹별 커버 여부 {group: bool}
        must_cover: 필수 커버 슬롯 목록

    Returns: decisive_evidence dict
    """
    support: list[dict] = []
    contra: list[dict] = []
    swing: list[dict] = []

    uncovered_slots = [s for s in must_cover if not coverage.get(s, False)]
    weakest_group = ""
    for g, covered in group_coverage.items():
        if not covered:
            weakest_group = g
            break
    if not weakest_group:
        # 모두 커버 → 가장 적은 support를 가진 그룹
        group_counts = {}
        for g, g_slots in COVERAGE_GROUPS.items():
            count = sum(1 for p in papers for s in p.get("claim_slots_supported", []) if s in g_slots)
            group_counts[g] = count
        if group_counts:
            weakest_group = min(group_counts, key=group_counts.get)

    for p in papers:
        role = p.get("evidence_role", "")
        slots = p.get("claim_slots_supported", [])
        score = _importance_score(p)
        dim = _dimension_for_slots(slots)

        entry = {
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", ""),
            "reason": p.get("why_selected", ""),
            "importance_score": score,
        }

        # contra: 가설을 위협하는 역할
        if role in _CONTRA_ROLES:
            entry["threatens_dimension"] = dim
            contra.append(entry)
        # support: 가설을 지지하는 역할
        elif role in _SUPPORT_ROLES:
            entry["supports_dimension"] = dim
            support.append(entry)

        # swing 감지 (3가지 경로)
        is_swing = False
        swing_reason = ""

        # 경로 1: uncovered 슬롯을 채울 수 있는 논문
        if any(s in uncovered_slots for s in slots):
            is_swing = True
            swing_reason = f"covers uncovered slot: {[s for s in slots if s in uncovered_slots]}"

        # 경로 2: semantic swing — support이면서 동시에 contra 역할 가능
        # (예: closest_prior_art가 mechanism을 지지하지만 novelty를 위협)
        if role in _CONTRA_ROLES and any(s in COVERAGE_GROUPS.get("novelty", set()) for s in slots):
            is_swing = True
            swing_reason = f"dual role: threatens {dim} but supports novelty slots"
        elif role in _SUPPORT_ROLES and any(s in {"falsification_criteria", "failure_modes"} for s in slots):
            is_swing = True
            swing_reason = f"dual role: supports {dim} but covers falsification/failure"

        # 경로 3: 높은 importance + weakest group 관련
        if score > 0.6 and weakest_group and dim == weakest_group:
            is_swing = True
            swing_reason = f"high importance ({score:.2f}) in weakest group ({weakest_group})"

        if is_swing:
            swing_entry = dict(entry)
            swing_entry["if_confirmed_changes"] = (
                "revise->approve" if score > 0.5 else "approve->revise"
            )
            swing_entry["swing_reason"] = swing_reason
            swing.append(swing_entry)

    # importance 순 정렬, 상위만 유지
    support.sort(key=lambda x: x["importance_score"], reverse=True)
    contra.sort(key=lambda x: x["importance_score"], reverse=True)
    swing.sort(key=lambda x: x["importance_score"], reverse=True)

    support = support[:3]
    contra = contra[:3]
    swing = swing[:2]

    # ── LLM 협업: semantic swing 정제 ──
    # GPT가 상위 논문들의 의미적 swing 가능성을 분석하고,
    # Gemini가 독립 검증하여 합의된 swing만 최종 채택
    swing = _llm_refine_swing(
        papers[:10], support, contra, swing, must_cover, uncovered_slots)

    # contra가 비어있으면 이유 설명
    if not contra:
        contra = [{"paper_id": "", "title": "",
                   "threatens_dimension": weakest_group or "unknown",
                   "reason": "No contra evidence found — all papers support the hypothesis. "
                             "This may indicate search bias or genuinely novel territory.",
                   "importance_score": 0.0}]

    # decision_pressure: 가장 약한 차원
    pressure_reason = ""
    if uncovered_slots:
        pressure_reason = f"Uncovered slots: {', '.join(uncovered_slots)}"
    elif weakest_group:
        pressure_reason = f"Weakest coverage group: {weakest_group}"
    else:
        pressure_reason = "All dimensions adequately covered"

    return {
        "decisive_evidence": {
            "support": support,
            "contra": contra,
            "swing": swing,
            "decision_pressure": {
                "primary_dimension": weakest_group or "none",
                "reason": pressure_reason,
            },
        }
    }


# ──────────────────────────────────────────────────────────
# LLM 협업 Swing 분석 (GPT 분석 → Gemini 검증)
# ──────────────────────────────────────────────────────────

def _llm_refine_swing(
    top_papers: list[dict],
    support: list[dict],
    contra: list[dict],
    rule_swing: list[dict],
    must_cover: list[str],
    uncovered_slots: list[str],
) -> list[dict]:
    """GPT가 의미적 swing을 분석하고 Gemini가 독립 검증한다.

    규칙 기반 swing을 출발점으로, LLM이 "이 논문의 결론이 확인되면
    가설 판정이 실제로 뒤집히는가?"를 의미적으로 판단한다.

    실패 시 규칙 기반 swing을 그대로 반환 (graceful fallback).
    """
    if not top_papers:
        return rule_swing

    # 상위 논문 요약 (토큰 절약)
    paper_summaries = []
    for p in top_papers[:8]:
        summary = {
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", ""),
            "evidence_role": p.get("evidence_role", ""),
            "claim_slots": p.get("claim_slots_supported", []),
            "abstract": (p.get("abstract") or "")[:300],
        }
        sections = p.get("sections", {})
        if sections.get("results"):
            summary["results_excerpt"] = sections["results"][:400]
        if sections.get("method"):
            summary["method_excerpt"] = sections["method"][:300]
        paper_summaries.append(summary)

    support_ids = [s.get("paper_id", "") for s in support]
    contra_ids = [c.get("paper_id", "") for c in contra if c.get("paper_id")]

    prompt = f"""You are a research evidence analyst. Identify SWING papers — papers whose
findings, if confirmed, would FLIP the hypothesis verdict (approve↔revise or revise↔reject).

## Current verdict signals
- Support papers: {json.dumps(support_ids)}
- Contra papers: {json.dumps(contra_ids)}
- Uncovered slots: {json.dumps(uncovered_slots)}
- Must cover: {json.dumps(must_cover)}

## Papers to analyze
{json.dumps(paper_summaries, ensure_ascii=False, indent=2)}

For each paper, ask: "If this paper's claims are WRONG or CONTRADICTED, does the hypothesis
verdict change?" If yes, it's a swing paper.

Return JSON only (no markdown):
{{
  "swing_papers": [
    {{
      "paper_id": "...",
      "if_confirmed_changes": "approve->revise | revise->approve | revise->reject",
      "swing_reason": "Why this paper's confirmation/refutation flips the verdict",
      "confidence": 0.0
    }}
  ]
}}"""

    # ── Step 1: GPT 분석 ──
    try:
        p_hash = prompt_hash(prompt)
        print(f"    [Swing / GPT] semantic swing 분석... (hash={p_hash})")
        client = get_openai_client()
        resp = llm_retry(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a research evidence analyst. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            label="GPT swing",
        )
        gpt_swings = json.loads(resp.choices[0].message.content).get("swing_papers", [])
    except Exception as e:
        print(f"    [Swing / GPT] 실패: {e} — 규칙 기반 fallback")
        return rule_swing

    if not gpt_swings:
        return rule_swing

    # ── Step 2: Gemini 독립 검증 (enriched output — GPT 결과를 보지 않음) ──
    gemini_prompt = f"""You are an independent research reviewer. For each paper below,
determine if it is a SWING paper — one whose confirmation/refutation would flip the
hypothesis verdict. Provide your own independent judgment.

## Papers
{json.dumps(paper_summaries, ensure_ascii=False, indent=2)}

## Uncovered evidence slots
{json.dumps(uncovered_slots)}

Return JSON only — for EACH swing paper provide full analysis:
{{
  "swing_papers": [
    {{
      "paper_id": "...",
      "if_confirmed_changes": "approve->revise | revise->approve | revise->reject",
      "swing_reason": "Why this paper flips the verdict",
      "confidence": 0.0
    }}
  ],
  "reasoning": "overall assessment"
}}"""

    gemini_swings: list[dict] = []
    try:
        print(f"    [Swing / Gemini] 독립 분석...")
        model = get_gemini_model()
        resp = llm_retry(model.generate_content, gemini_prompt, label="Gemini swing")
        text = resp.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        gemini_result = json.loads(text)
        gemini_swings = gemini_result.get("swing_papers", [])
        print(f"    [Swing / Gemini] {len(gemini_swings)}개 swing 감지")
    except Exception as e:
        print(f"    [Swing / Gemini] 실패: {e} — GPT 결과만 사용")
        gemini_swings = []

    # Gemini paper_id 집합
    gemini_swing_map = {s.get("paper_id", ""): s for s in gemini_swings}
    gemini_confirmed_ids = set(gemini_swing_map.keys())
    # fallback: Gemini 실패 시 GPT 결과 전체 수용
    if not gemini_swings:
        gemini_confirmed_ids = {s.get("paper_id", "") for s in gpt_swings}

    # ── Step 3: 합의 — GPT + Gemini 양측 분석 결합 ──
    final_swing: list[dict] = []
    for gs in gpt_swings:
        pid = gs.get("paper_id", "")
        if pid in gemini_confirmed_ids:
            orig = next((p for p in top_papers if p.get("paper_id") == pid), {})
            gem = gemini_swing_map.get(pid, {})

            # 합의 수준 판정
            gpt_effect = gs.get("if_confirmed_changes", "")
            gem_effect = gem.get("if_confirmed_changes", gpt_effect)
            if gpt_effect == gem_effect:
                consensus_level = "strong"
                final_effect = gpt_effect
            elif gem.get("paper_id"):
                consensus_level = "weak"
                final_effect = gpt_effect  # GPT 우선, 불일치 기록
            else:
                consensus_level = "rule_fallback"
                final_effect = gpt_effect

            final_swing.append({
                "paper_id": pid,
                "title": orig.get("title", ""),
                "gpt_if_confirmed_changes": gpt_effect,
                "gemini_if_confirmed_changes": gem_effect,
                "final_if_confirmed_changes": final_effect,
                "gpt_reason": gs.get("swing_reason", ""),
                "gemini_reason": gem.get("swing_reason", ""),
                "final_reason": f"[{consensus_level}] {gs.get('swing_reason', '')}",
                "consensus_level": consensus_level,
                "importance_score": _importance_score(orig) if orig else 0.5,
                "confidence": (gs.get("confidence", 0.5) + gem.get("confidence", 0.5)) / 2,
            })

    # GPT+Gemini 합의 swing이 있으면 사용, 없으면 규칙 기반 유지
    if final_swing:
        # 규칙 기반 swing 중 LLM에서 확인되지 않은 것도 보존 (낮은 우선순위)
        for rs in rule_swing:
            if rs.get("paper_id") not in {f["paper_id"] for f in final_swing}:
                rs["swing_reason"] = f"[rule-based] {rs.get('swing_reason', '')}"
                final_swing.append(rs)
        print(f"    [Swing] 최종: {len(final_swing)}개 (LLM 합의 {len([f for f in final_swing if '[LLM' in f.get('swing_reason', '')])}개)")
        return final_swing[:3]
    else:
        print(f"    [Swing] LLM 합의 없음 — 규칙 기반 {len(rule_swing)}개 유지")
        return rule_swing
