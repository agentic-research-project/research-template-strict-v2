"""
Scientific Betting Engine

희소한 증거에서 "exploit / probe / reject" 중 어떤 전략이 최선인지를 계산한다.
- exploit: 증거가 강하고 리스크 낮음 → 밀어붙이기
- probe: 증거 희소하지만 정보이득 큼 → 소규모 검증 먼저
- reject: contra 증거 강하고 downside 큼 → 포기/전환

산출물: experiments/{slug}/reports/scientific_bet.json

사용법:
    from lab.scientific_betting import compute_scientific_bet
    bet = compute_scientific_bet(evidence_coverage, decisive_evidence, ...)
"""


def _count_uncovered(coverage: dict[str, bool]) -> int:
    """미충족 슬롯 수."""
    return sum(1 for v in coverage.values() if not v)


def _avg_importance(items: list[dict]) -> float:
    """importance_score 평균."""
    scores = [i.get("importance_score", 0) for i in items]
    return sum(scores) / len(scores) if scores else 0.0


def compute_scientific_bet(
    evidence_coverage: dict,
    decisive_evidence: dict,
    constraints_structured: dict,
    falsification_criteria: list,
    validator_summary: dict | None = None,
) -> dict:
    """과학적 베팅을 계산한다.

    Args:
        evidence_coverage: papers.json의 evidence_coverage
        decisive_evidence: decisive_evidence.json의 decisive_evidence
        constraints_structured: topic_analysis.json의 constraints_structured
        falsification_criteria: hypothesis.json의 falsification_criteria
        validator_summary: validation.json의 summary (없으면 pre-validation)

    Returns: scientific_bet dict (bet_type, bet_confidence, info_gain, downside, ...)
    """
    coverage = evidence_coverage.get("coverage", {})
    sufficient = evidence_coverage.get("sufficient", False)
    group_cov = evidence_coverage.get("group_coverage", {})

    support = decisive_evidence.get("support", [])
    contra = decisive_evidence.get("contra", [])
    swing = decisive_evidence.get("swing", [])
    pressure = decisive_evidence.get("decision_pressure", {})

    # ── 1. expected_information_gain ──
    # 미충족 슬롯이 많을수록 + swing이 강할수록 → 정보이득 큼
    n_uncovered = _count_uncovered(coverage)
    swing_importance = _avg_importance(swing)
    n_falsif = len(falsification_criteria)
    # 0-1 범위
    info_gain = round(min(1.0,
        n_uncovered * 0.15 +
        swing_importance * 0.30 +
        (0.2 if n_falsif >= 2 else 0.05) +
        (0.15 if not sufficient else 0.0)
    ), 3)

    # ── 2. downside_if_wrong (Expected Loss 기반) ──
    # E[Loss] = P(wrong) × Impact(wrong)
    # P(wrong) ∝ contra 강도 + uncovered 비율
    # Impact(wrong) ∝ constraint 엄격도 + feasibility 부족
    contra_importance = _avg_importance(contra)
    cs = constraints_structured or {}

    # P(wrong): contra가 강하고 coverage가 낮을수록 높음
    total_slots = max(len(coverage), 1)
    uncovered_ratio = n_uncovered / total_slots
    p_wrong = min(1.0, contra_importance * 0.5 + uncovered_ratio * 0.3 +
                  (0.2 if not sufficient else 0.0))

    # Impact(wrong): constraint가 엄격할수록 실패 비용 큼
    constraint_severity = 0.0
    if cs.get("param_budget_M") and cs["param_budget_M"] < 1.0:
        constraint_severity += 0.25
    if cs.get("single_gpu"):
        constraint_severity += 0.1
    if cs.get("latency_sensitive"):
        constraint_severity += 0.2
    if cs.get("no_pretrained"):
        constraint_severity += 0.1
    if not group_cov.get("feasibility", True):
        constraint_severity += 0.2
    impact_wrong = min(1.0, constraint_severity + 0.1)  # base cost 0.1

    # Expected Loss = P(wrong) × Impact(wrong)
    downside = round(min(1.0, p_wrong * impact_wrong +
        (0.15 if n_uncovered >= 3 else 0.0)  # 대규모 미충족 보너스
    ), 3)

    # ── 3. bet_confidence (validator 반영) ──
    if validator_summary:
        avg_score = validator_summary.get("average_score", 5.0)
        bet_confidence = round(avg_score / 10.0, 3)
    else:
        # pre-validation: coverage 기반 추정
        covered_ratio = sum(1 for v in coverage.values() if v) / max(len(coverage), 1)
        support_strength = _avg_importance(support)
        bet_confidence = round(covered_ratio * 0.5 + support_strength * 0.5, 3)

    # ── 4. bet_type 결정 (deterministic) ──
    if bet_confidence >= 0.7 and contra_importance < 0.3 and downside < 0.4:
        bet_type = "exploit"
    elif contra_importance >= 0.6 and downside >= 0.6:
        bet_type = "reject"
    else:
        bet_type = "probe"

    # ── 5. dominant_uncertainty ──
    if not group_cov.get("novelty", True):
        dominant = "novelty"
    elif not group_cov.get("validity", True):
        dominant = "mechanism"
    elif not group_cov.get("feasibility", True):
        dominant = "feasibility"
    elif n_falsif < 1:
        dominant = "falsification"
    else:
        dominant = pressure.get("primary_dimension", "none")

    # ── 6. minimal_test_to_disconfirm ──
    tests: list[str] = []
    if dominant == "mechanism":
        tests.append("Run mechanism-removal ablation (remove core component, compare metric)")
    if dominant == "novelty":
        tests.append("Compare against closest prior art under identical constraints")
    if dominant == "feasibility":
        tests.append("Verify param count and inference latency under target constraints")
    if dominant == "falsification":
        tests.append("Define explicit metric threshold for failure (e.g., accuracy < baseline - 2%)")
    if not tests:
        tests.append("Run baseline comparison with fixed parameter budget")
    if swing:
        top_swing = swing[0]
        tests.append(f"Verify swing evidence: {top_swing.get('title', top_swing.get('paper_id', ''))}")

    # ── 7. rationale ──
    rationale_parts = []
    if bet_type == "exploit":
        rationale_parts.append(f"Strong support ({_avg_importance(support):.2f}), weak contra ({contra_importance:.2f})")
    elif bet_type == "reject":
        rationale_parts.append(f"Strong contra ({contra_importance:.2f}), high downside ({downside:.2f})")
    else:
        rationale_parts.append(f"Mixed evidence (info_gain={info_gain:.2f}), moderate risk")
    rationale_parts.append(f"Dominant uncertainty: {dominant}")

    return {
        "scientific_bet": {
            "bet_type": bet_type,
            "bet_confidence": bet_confidence,
            "expected_information_gain": info_gain,
            "downside_if_wrong": downside,
            "minimal_test_to_disconfirm": tests[:3],
            "rationale": ". ".join(rationale_parts),
            "dominant_uncertainty": dominant,
        }
    }
