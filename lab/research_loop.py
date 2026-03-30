"""
Stage 8: Research Loop — 실험 실행 + Path A/B/C revision 루프

experiments/{slug}/runs/v{N}/ 패키지를 실행하고 결과를 분석한다.
목표 미달이면 Path A (코드 수정) 로 최대 max_rounds회 재시도한다.
Path B/C가 필요하면 revision_request.json을 생성하고 루프를 종료한다.

분석 파이프라인 (Multi-model, consensus-locked):
  GPT    -> 주요 해석자 (심층 결과 해석, suggested_path + evidence)
  Gemini -> 독립 2차 진단 (short diagnosis, agreement_with_gpt)
  합의 레이어 -> GPT+Gemini 통합, consensus_path_candidate 생성
  GPT    -> Path A 시 구현 패치 제안 (patch-only, 결정 권한 없음)
  Claude -> override reviewer (합의 경로 기본, escalation만 override 허용)
  사후검증 -> override 최종 허가자 (guardrail 위반 시 candidate_path로 복원)

Runner 추상화:
  --runner-type github  (기본값) GitHub Actions 워크플로우 트리거 + 결과 수집
  --runner-type local   로컬 subprocess 실행

사용법:
  python -m lab.research_loop \\
    --pkg-dir         experiments/{slug}/runs/v1 \\
    --topic-file      experiments/{slug}/reports/topic_analysis.json \\
    --hypothesis-file experiments/{slug}/reports/hypothesis.json \\
    --code-file       experiments/{slug}/reports/code_analysis.json \\
    [--max-rounds 3] [--runner-type github]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from lab.config import (
    query_claude, parse_json, get_openai_client, get_gemini_model,
    OPENAI_MODEL, GEMINI_MODEL,
    result_version_dir, results_dir, reports_dir, slug_from_pkg, version_from_pkg,
    # Diagnostic engine constants
    CONVERGENCE_TAIL_FRACTION, CONVERGENCE_REL_THRESHOLD,
    PLATEAU_WINDOW, PLATEAU_REL_THRESHOLD,
    OVERFIT_GAP_THRESHOLD, UNDERTRAINING_IMPROVEMENT_FACTOR,
    DIVERGENCE_LOSS_RATIO, ATTAINMENT_FAILURE_RATIO,
    PATH_B_CONSECUTIVE_PLATEAUS, PATH_C_MIN_CONTRADICTIONS, PATH_C_MIN_RUNS,
    CONFIDENCE_WEIGHTS,
    ABLATION_EFFECT_THRESHOLD,
    EFFECT_SIZE_SMALL, EFFECT_SIZE_MEDIUM, EFFECT_SIZE_LARGE,
    METRIC_VALID_RANGES,
    # LLM stability
    llm_retry, prompt_hash,
    # Consensus strength
    CONFIDENCE_LEVEL_MAP, PATH_DISTANCE,
    CONSENSUS_STRENGTH_PATH_B, CONSENSUS_STRENGTH_PATH_C,
)
from lab.model_generator import generate_experiment_package, _save_proposal, _archive_proposal
from lab.runners import BaseRunner, create_runner


# ──────────────────────────────────────────────────────────
# 패키지 구조 검증
# ──────────────────────────────────────────────────────────

REQUIRED_FILES = [
    "train.py", "model.py", "module.py", "data.py",
    "configs/default.yaml", "configs/fast.yaml",
    "scripts/smoke_test.py",
]

def _validate_package(pkg_dir: Path) -> list[str]:
    return [f for f in REQUIRED_FILES if not (pkg_dir / f).exists()]


# ──────────────────────────────────────────────────────────
# Runner 준비 상태 확인 (루프 진입 전)
# ──────────────────────────────────────────────────────────

def _check_runner_ready(runner: BaseRunner) -> None:
    """runner의 is_ready() capability check로 실행 가능 여부를 확인한다.

    side-effect 없이 즉시 완료된다 (run_smoke 등 실제 실행 없음).
    준비되지 않은 runner이면 structured error를 출력하고 sys.exit(1)로 종료한다.
    """
    ready, reason = runner.is_ready()
    if not ready:
        error_info = {
            "error":        "runner_not_ready",
            "runner_type":  runner.__class__.__name__,
            "reason":       reason,
            "suggestion":   (
                "GitHubActionsRunner 사용 시 필요한 환경변수: "
                "GITHUB_TOKEN (workflow+contents scope), GITHUB_OWNER, GITHUB_REPO. "
                "GitHub Secrets 필요: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY. "
                "self-hosted GPU runner (runs-on: [self-hosted, gpu]) 필요. "
                "또는 --runner-type local 로 변경하세요."
            ),
        }
        print("\n[FATAL] Runner 준비 안 됨 — 실험 루프 진입 중단")
        print(json.dumps(error_info, ensure_ascii=False, indent=2))
        sys.exit(1)


# ──────────────────────────────────────────────────────────
# hypothesis implementation audit 결과 로드
# ──────────────────────────────────────────────────────────

def _load_hypothesis_audit(pkg_dir: Path) -> dict:
    """패키지의 artifacts/에서 hypothesis implementation audit 결과를 로드한다.

    audit 파일이 없으면 빈 dict를 반환한다 (이전 버전 호환).
    """
    audit = {}
    artifacts_dir = pkg_dir / "artifacts"
    for key, filename in [
        ("mechanism_audit",    "mechanism_audit.json"),
        ("metric_audit",       "metric_audit.json"),
        ("constraints_audit",  "constraints_audit.json"),
    ]:
        fpath = artifacts_dir / filename
        if fpath.exists():
            try:
                audit[key] = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception:
                audit[key] = {}
    return audit


# ──────────────────────────────────────────────────────────
# Diagnostic Helpers (A-1 ~ A-6)
# ──────────────────────────────────────────────────────────

# --- stdout / metrics trace parsing ---

_RE_EPOCH_LOSS = re.compile(
    r"(?:epoch|ep)\s*[\[:\s]*(\d+).*?"
    r"(?:loss|train_loss)\s*[:=]\s*([\d.eE+\-]+)",
    re.IGNORECASE,
)
_RE_VAL_METRIC = re.compile(
    r"(?:val|valid|test)[\s_]*(?:loss|acc|psnr|ssim|f1|metric)\s*[:=]\s*([\d.eE+\-]+)",
    re.IGNORECASE,
)
_RE_LR = re.compile(
    r"(?:lr|learning_rate)\s*[:=]\s*([\d.eE+\-]+)",
    re.IGNORECASE,
)


def _parse_traces_from_stdout(stdout_lines: list[str]) -> dict:
    """Parse epoch-level train loss, val metric, and lr traces from stdout lines.

    Returns dict with keys: train_losses, val_metrics, lr_values (all list[float]).
    """
    train_losses: list[float] = []
    val_metrics: list[float] = []
    lr_values: list[float] = []
    for line in stdout_lines:
        m = _RE_EPOCH_LOSS.search(line)
        if m:
            try:
                train_losses.append(float(m.group(2)))
            except (ValueError, OverflowError):
                pass
        m = _RE_VAL_METRIC.search(line)
        if m:
            try:
                val_metrics.append(float(m.group(1)))
            except (ValueError, OverflowError):
                pass
        m = _RE_LR.search(line)
        if m:
            try:
                lr_values.append(float(m.group(1)))
            except (ValueError, OverflowError):
                pass
    return {
        "train_losses": train_losses,
        "val_metrics": val_metrics,
        "lr_values": lr_values,
    }


def _parse_traces_from_metrics(metrics: dict) -> dict:
    """Extract trace data from final_metrics.json rich fields if present.

    Rich trace fields (trainer가 가능하면 기록):
        train_loss_history_tail    — 마지막 N epoch의 train loss 리스트
        val_metric_history_tail    — 마지막 N epoch의 val metric 리스트
        lr_history_tail            — 마지막 N epoch의 learning rate 리스트
        train_metric_history_tail  — 마지막 N epoch의 train metric 리스트
        train_loss_last            — 최종 epoch train loss (float)
        train_loss_best            — 최소 train loss (float)
        best_epoch                 — val metric 최고 epoch (int)
        early_stop_triggered       — early stopping 발동 여부 (bool)
    """
    return {
        "train_loss_last":            metrics.get("train_loss_last"),
        "train_loss_best":            metrics.get("train_loss_best"),
        "train_loss_history_tail":    metrics.get("train_loss_history_tail"),
        "val_metric_history_tail":    metrics.get("val_metric_history_tail"),
        "lr_history_tail":            metrics.get("lr_history_tail"),
        "train_metric_history_tail":  metrics.get("train_metric_history_tail"),
        "best_epoch":                 metrics.get("best_epoch"),
        "early_stop_triggered":       metrics.get("early_stop_triggered"),
    }


# --- A-2: Training Stability ---
# 임계값은 lab/config.py에서 관리 (CONVERGENCE_*, PLATEAU_*, OVERFIT_*, etc.)


def _infer_training_stability(
    run_result: dict,
    metrics: dict,
) -> dict:
    """A-2: Compute real training stability diagnostics from traces and metrics.

    Returns dict matching training_stability schema with:
        nan_detected, loss_converged, lr_schedule_ok,
        plateau_detected, overfitting_suspected, undertraining_suspected, notes
    """
    stdout_lines = run_result.get("stdout_lines", [])
    stdout_text = "\n".join(stdout_lines)
    status = run_result.get("status", "")

    # --- NaN / Inf detection ---
    nan_detected = bool(
        re.search(r"\bnan\b", stdout_text, re.IGNORECASE)
        or re.search(r"\binf\b", stdout_text, re.IGNORECASE)
    )

    # --- Parse traces ---
    stdout_traces = _parse_traces_from_stdout(stdout_lines)
    metrics_traces = _parse_traces_from_metrics(metrics)

    train_losses = stdout_traces["train_losses"]
    val_metrics = stdout_traces["val_metrics"]
    lr_values = stdout_traces["lr_values"]

    # Override with richer metrics-json traces if available (regex fallback보다 우선)
    if metrics_traces.get("train_loss_history_tail"):
        train_losses = [float(v) for v in metrics_traces["train_loss_history_tail"]]
    if metrics_traces.get("val_metric_history_tail"):
        val_metrics = [float(v) for v in metrics_traces["val_metric_history_tail"]]
    if metrics_traces.get("lr_history_tail"):
        lr_values = [float(v) for v in metrics_traces["lr_history_tail"]]

    notes: list[str] = []

    # --- Loss convergence ---
    loss_converged = False
    if train_losses and len(train_losses) >= 2:
        tail_start = max(1, len(train_losses) - max(2, int(len(train_losses) * CONVERGENCE_TAIL_FRACTION)))
        tail = train_losses[tail_start:]
        if tail and tail[0] != 0:
            rel_change = abs(tail[-1] - tail[0]) / (abs(tail[0]) + 1e-12)
            loss_converged = rel_change < CONVERGENCE_REL_THRESHOLD
        if not loss_converged and len(train_losses) > 3:
            notes.append("convergence:loss_still_changing")
    elif status == "success" and bool(metrics):
        # Fallback: if we got successful metrics but no loss trace, assume converged
        loss_converged = True
    if nan_detected:
        loss_converged = False
        notes.append("nan:detected_in_output")

    # --- Early divergence ---
    if train_losses and len(train_losses) >= 3:
        if train_losses[-1] > train_losses[0] * DIVERGENCE_LOSS_RATIO:
            notes.append("divergence:final_loss_gt_2x_initial")
            loss_converged = False

    # --- Plateau detection ---
    plateau_detected = False
    plateau_epoch = None
    if val_metrics and len(val_metrics) >= PLATEAU_WINDOW:
        for i in range(len(val_metrics) - PLATEAU_WINDOW + 1):
            window = val_metrics[i:i + PLATEAU_WINDOW]
            if window[0] != 0:
                max_rel_change = max(
                    abs(window[j + 1] - window[j]) / (abs(window[0]) + 1e-12)
                    for j in range(len(window) - 1)
                )
                if max_rel_change < PLATEAU_REL_THRESHOLD:
                    plateau_detected = True
                    plateau_epoch = i + 1
                    break
        if plateau_detected and plateau_epoch is not None:
            notes.append(f"plateau:epoch{plateau_epoch}")
    elif train_losses and len(train_losses) >= PLATEAU_WINDOW and not val_metrics:
        # Use train loss as proxy if no val metrics
        for i in range(len(train_losses) - PLATEAU_WINDOW + 1):
            window = train_losses[i:i + PLATEAU_WINDOW]
            if window[0] != 0:
                max_rel_change = max(
                    abs(window[j + 1] - window[j]) / (abs(window[0]) + 1e-12)
                    for j in range(len(window) - 1)
                )
                if max_rel_change < PLATEAU_REL_THRESHOLD:
                    plateau_detected = True
                    plateau_epoch = i + 1
                    notes.append(f"plateau:loss_epoch{plateau_epoch}")
                    break

    # --- LR schedule check ---
    lr_schedule_ok: bool | None = None  # None = not applicable / no data
    if lr_values and len(lr_values) >= 2:
        unique_lrs = set(round(v, 10) for v in lr_values)
        if len(unique_lrs) > 1:
            lr_schedule_ok = True  # LR changed during training -> scheduler active
        else:
            lr_schedule_ok = False
            notes.append("scheduler:constant_lr")
    else:
        notes.append("scheduler:not_applicable")

    # --- Overfitting detection ---
    overfitting_suspected = False
    train_loss_last = metrics_traces["train_loss_last"]
    train_metric_tail = metrics_traces.get("train_metric_history_tail", [])

    # Method 1: train_loss_last vs val_metric gap (OVERFIT_GAP_THRESHOLD 사용)
    if train_loss_last is not None and val_metrics and len(val_metrics) >= 2:
        # 낮은 train loss + val metric 정체 → overfitting
        val_best = max(val_metrics) if val_metrics else 0
        val_last = val_metrics[-1] if val_metrics else 0
        if val_best > 0 and (val_best - val_last) / (val_best + 1e-12) > OVERFIT_GAP_THRESHOLD:
            overfitting_suspected = True
            notes.append(f"overfit:val_degradation (best={val_best:.4f} → last={val_last:.4f}, gap>{OVERFIT_GAP_THRESHOLD:.0%})")

    # Method 2: train metric vs val metric gap
    if train_metric_tail and val_metrics and len(train_metric_tail) >= 2 and len(val_metrics) >= 2:
        train_last = float(train_metric_tail[-1])
        val_last_m = float(val_metrics[-1])
        if train_last > 0:
            gap = (train_last - val_last_m) / (train_last + 1e-12)
            if gap > OVERFIT_GAP_THRESHOLD:
                overfitting_suspected = True
                if "overfit:" not in " ".join(notes):
                    notes.append(f"overfit:train_val_gap (train={train_last:.4f}, val={val_last_m:.4f}, gap={gap:.2%})")

    # Method 3: train loss decreasing but val metrics stagnant
    if train_losses and val_metrics and len(train_losses) >= 3 and len(val_metrics) >= 3:
        train_improving = train_losses[-1] < train_losses[0] * 0.8  # 20% improvement
        mid_idx = max(1, len(val_metrics) // 2)
        val_stagnant = abs(val_metrics[-1] - val_metrics[mid_idx]) / (abs(val_metrics[0]) + 1e-12) < PLATEAU_REL_THRESHOLD * 2
        if train_improving and val_stagnant and not overfitting_suspected:
            overfitting_suspected = True
            notes.append("overfit:train_loss_80pct_improved_but_val_stagnant")

    # --- Undertraining detection ---
    undertraining_suspected = False
    if loss_converged is False and not nan_detected and status == "success":
        undertraining_suspected = True
        notes.append("underfit:loss_not_converged_no_nan")
    elif train_losses and len(train_losses) >= 2:
        # Still improving significantly in last epochs
        if len(train_losses) >= 3:
            last_improvement = abs(train_losses[-1] - train_losses[-2]) / (abs(train_losses[-2]) + 1e-12)
            if last_improvement > CONVERGENCE_REL_THRESHOLD * UNDERTRAINING_IMPROVEMENT_FACTOR:
                undertraining_suspected = True
                if "underfit:" not in " ".join(notes):
                    notes.append("underfit:still_improving_rapidly")

    return {
        "loss_converged": loss_converged,
        "nan_detected": nan_detected,
        "lr_schedule_ok": lr_schedule_ok,
        "plateau_detected": plateau_detected,
        "overfitting_suspected": overfitting_suspected,
        "undertraining_suspected": undertraining_suspected,
        "notes": notes,
    }


# --- A-3: Bottleneck Candidates ---

def _infer_bottlenecks(
    stability: dict,
    primary_metric: dict,
    hypothesis_impl: dict,
    run_result: dict,
) -> list[dict]:
    """A-3: Deterministic rule-based bottleneck candidate generation.

    Returns list of {name, severity, evidence}.
    """
    candidates: list[dict] = []
    target = float(primary_metric.get("target", 0))
    value = float(primary_metric.get("value", 0))
    met = primary_metric.get("met", False)
    status = run_result.get("status", "")

    nan_detected = stability.get("nan_detected", False)
    loss_converged = stability.get("loss_converged", False)
    plateau_detected = stability.get("plateau_detected", False)
    overfitting = stability.get("overfitting_suspected", False)
    undertraining = stability.get("undertraining_suspected", False)

    mech_audit = hypothesis_impl.get("mechanism_audit", {})
    metric_audit = hypothesis_impl.get("metric_audit", {})
    constraints_audit = hypothesis_impl.get("constraints_audit", {})

    attainment_ratio = value / target if target > 0 else 0

    # Rule: NaN → optimization bottleneck
    if nan_detected:
        candidates.append({
            "name": "optimization bottleneck",
            "severity": "critical",
            "confidence": 0.95,
            "evidence": ["nan_detected=true", "possible optimizer/lr/gradient issue"],
        })

    # Rule: metric < 50% of target AND loss not converged → execution failure
    if attainment_ratio < ATTAINMENT_FAILURE_RATIO and not loss_converged and not met:
        candidates.append({
            "name": "execution failure",
            "severity": "critical",
            "confidence": 0.85,
            "evidence": [
                f"attainment_ratio={attainment_ratio:.2f} (<{ATTAINMENT_FAILURE_RATIO})",
                "loss_converged=false",
                "core implementation mismatch likely",
            ],
        })

    # Rule: plateau → representation bottleneck
    if plateau_detected and not met:
        candidates.append({
            "name": "representation bottleneck",
            "severity": "medium",
            "confidence": 0.70,
            "evidence": ["metric plateau", "stable training", f"attainment_ratio={attainment_ratio:.2f}"],
        })

    # Rule: overfitting → regularization gap
    if overfitting:
        candidates.append({
            "name": "regularization gap",
            "severity": "medium",
            "confidence": 0.75,
            "evidence": ["train improving but val stagnant/degrading", "regularization or data split issue"],
        })

    # Rule: mechanism_ok=false → core mechanism not implemented
    if mech_audit and not mech_audit.get("implemented", True):
        candidates.append({
            "name": "core mechanism not implemented",
            "severity": "critical",
            "confidence": 0.90,
            "evidence": ["mechanism_audit.implemented=false"] + mech_audit.get("missing_links", [])[:2],
        })

    # Rule: metric_ok=false → metrics contract failure
    if metric_audit and not metric_audit.get("implemented", True):
        candidates.append({
            "name": "metrics contract failure",
            "severity": "high",
            "confidence": 0.90,
            "evidence": [
                "metric_audit.implemented=false",
                f"expected={metric_audit.get('primary_metric_expected', '?')}",
                f"found={metric_audit.get('primary_metric_found', '?')}",
            ],
        })

    # Rule: constraints_ok=false → constraint violation
    if constraints_audit and not constraints_audit.get("implemented", True):
        violations = constraints_audit.get("violations", [])
        candidates.append({
            "name": "constraint violation",
            "severity": "high",
            "confidence": 0.85,
            "evidence": ["constraints_audit.implemented=false"] + violations[:2],
        })

    # Rule: execution failure (smoke/timeout/parse error)
    if status in ("smoke_failed", "failed", "metrics_parse_error", "timeout"):
        # execution failure가 아직 없으면 추가
        if not any(c["name"] == "execution failure" for c in candidates):
            candidates.append({
                "name": "execution failure",
                "severity": "critical",
                "confidence": 0.95,
                "evidence": [f"status={status}"],
            })

    # Rule: undertraining → optimization bottleneck (mild)
    if undertraining and not met:
        candidates.append({
            "name": "optimization bottleneck",
            "severity": "low",
            "confidence": 0.50,
            "evidence": ["loss not fully converged", "more epochs or lr tuning may help"],
        })

    # ── Family-specific bottleneck ──
    # experiment_spec에서 task_family를 읽어 family-aware 진단 추가
    task_family = run_result.get("metadata", {}).get("task_family", "")
    if not task_family:
        # spec에서 직접 읽기 시도
        spec_meta = run_result.get("spec", {})
        task_family = spec_meta.get("task_family", "")

    _FAMILY_BOTTLENECKS = {
        "detection": [
            ("bbox head mismatch", "bbox format or head output mismatch"),
            ("NMS/postprocess failure", "postprocess pipeline broken or missing"),
            ("eval contract failure", "mAP/AP50 metric not computed correctly"),
        ],
        "representation_learning": [
            ("embedding collapse", "embeddings converge to single point"),
            ("projection head mismatch", "projection head missing or misconfigured"),
            ("probe evaluation missing", "no linear probe evaluation downstream"),
        ],
        "few_shot_learning": [
            ("episodic sampler failure", "non-episodic dataloader used"),
            ("support/query leakage", "support and query sets overlap"),
            ("distance metric mismatch", "wrong distance metric for prototype/relation"),
        ],
        "generation": [
            ("mode collapse", "generator produces limited variety"),
            ("training instability", "GAN oscillation or diffusion divergence"),
        ],
        "contrastive_learning": [
            ("representation collapse", "all embeddings map to same vector"),
            ("augmentation weakness", "insufficient augmentation diversity"),
        ],
        "anomaly_detection": [
            ("anomaly in training", "training data contaminated with anomalies"),
            ("score collapse", "anomaly scores not discriminative"),
        ],
        "physics_informed": [
            ("PDE residual not converging", "physics loss not decreasing — check collocation/activation"),
            ("boundary condition violation", "boundary loss high — insufficient boundary points or wrong BC"),
            ("spectral bias", "network captures low frequencies only — use Fourier features or SIREN"),
        ],
    }

    family_bns = _FAMILY_BOTTLENECKS.get(task_family, [])
    if family_bns and not met:
        # family-specific bottleneck을 soft candidate로 추가
        for bn_name, bn_detail in family_bns[:2]:
            candidates.append({
                "name": f"[{task_family}] {bn_name}",
                "severity": "medium",
                "confidence": 0.40,
                "evidence": [bn_detail, f"family-specific risk for {task_family}"],
            })

    # Ensure at least 1 candidate if target not met
    if not candidates and not met:
        candidates.append({
            "name": "representation bottleneck",
            "severity": "low",
            "confidence": 0.30,
            "evidence": [f"target not met (attainment={attainment_ratio:.2f}) but no specific issue detected"],
        })

    return candidates


# --- A-4: Recommended Next Actions ---

def _infer_recommended_actions(
    stability: dict,
    primary_metric: dict,
    hypothesis_impl: dict,
    bottleneck_candidates: list[dict],
    run_history: list[dict],
    run_result: dict,
) -> list[dict]:
    """A-4: Rule-based recommended_next_actions as deterministic prior.

    Returns list of {path, priority, rationale, evidence}.
    """
    actions: list[dict] = []
    met = primary_metric.get("met", False)
    status = run_result.get("status", "")

    mech_audit = hypothesis_impl.get("mechanism_audit", {})
    metric_audit = hypothesis_impl.get("metric_audit", {})
    constraints_audit = hypothesis_impl.get("constraints_audit", {})

    # Rule: target met -> done
    if met:
        actions.append({
            "path": "done",
            "priority": "high",
            "rationale": "target metric achieved",
            "evidence": [f"primary_metric.met=true"],
        })
        return actions

    # Rule: execution failure -> Path A
    if status in ("smoke_failed", "failed", "metrics_parse_error", "timeout"):
        actions.append({
            "path": "A",
            "priority": "high",
            "rationale": f"execution failure ({status}) requires implementation fix",
            "evidence": [f"status={status}"],
        })

    # Rule: mechanism_ok=false -> Path A high
    if mech_audit and not mech_audit.get("implemented", True):
        actions.append({
            "path": "A",
            "priority": "high",
            "rationale": "core mechanism not implemented",
            "evidence": ["mechanism_audit.implemented=false"],
        })

    # Rule: metric_ok=false -> Path A high
    if metric_audit and not metric_audit.get("correct", True):
        actions.append({
            "path": "A",
            "priority": "high",
            "rationale": "evaluation/metrics contract broken",
            "evidence": ["metric_audit.correct=false"],
        })

    # Rule: constraints_ok=false -> Path A high
    if constraints_audit and not constraints_audit.get("satisfied", True):
        actions.append({
            "path": "A",
            "priority": "high",
            "rationale": "constraint violation requires fix",
            "evidence": ["constraints_audit.satisfied=false"],
        })

    # Rule: NaN -> Path A
    if stability.get("nan_detected"):
        actions.append({
            "path": "A",
            "priority": "high",
            "rationale": "NaN/Inf detected, training instability",
            "evidence": ["nan_detected=true"],
        })

    # Rule: 2+ consecutive plateaus -> Path B medium
    n_runs = len(run_history)
    consecutive_plateaus = 0
    for entry in reversed(run_history):
        entry_stability = entry.get("training_stability", {})
        if entry_stability.get("plateau_detected", False):
            consecutive_plateaus += 1
        else:
            break
    if stability.get("plateau_detected"):
        consecutive_plateaus += 1  # current run (not yet in history when called during build)

    if consecutive_plateaus >= PATH_B_CONSECUTIVE_PLATEAUS:
        actions.append({
            "path": "B",
            "priority": "medium",
            "rationale": f"{consecutive_plateaus} consecutive plateaus suggest hypothesis refinement needed",
            "evidence": [f"consecutive_plateaus={consecutive_plateaus}"],
        })

    # Rule: 3+ runs + mechanism implemented + repeated contradiction -> Path C
    mechanism_implemented = not mech_audit or mech_audit.get("implemented", True)
    if n_runs >= PATH_C_MIN_RUNS and mechanism_implemented:
        # Check for "repeated contradiction": metric consistently far from target
        contradiction_count = sum(
            1 for entry in run_history
            if entry.get("metrics", {}).get("met") is False
            and entry.get("training_stability", {}).get("loss_converged", False)
        )
        if contradiction_count >= PATH_C_MIN_CONTRADICTIONS:
            actions.append({
                "path": "C",
                "priority": "high",
                "rationale": "3+ runs with mechanism implemented but target consistently unmet despite stable training",
                "evidence": [
                    f"contradiction_count={contradiction_count}",
                    "mechanism_implemented=true",
                    "repeated experimental contradiction",
                ],
            })

    # Fallback: if no actions yet and not met, suggest Path A
    if not actions:
        actions.append({
            "path": "A",
            "priority": "medium",
            "rationale": "target not met, implementation improvement suggested",
            "evidence": [f"attainment={primary_metric.get('value', 0)}/{primary_metric.get('target', 0)}"],
        })

    return actions


# --- A-5: Confidence Model ---

# Confidence weights: lab/config.py CONFIDENCE_WEIGHTS 사용


def _estimate_confidence(
    primary_metric: dict,
    stability: dict,
    hypothesis_impl: dict,
    deltas: dict,
    run_history: list[dict],
    run_result: dict,
) -> dict:
    """A-5: Weighted confidence score model with explanation.

    Returns dict with sub-scores, final_confidence, and explanation.
    Also includes backward-compatible flat 'confidence' float.
    """
    target = float(primary_metric.get("target", 0))
    value = float(primary_metric.get("value", 0))
    met = primary_metric.get("met", False)
    status = run_result.get("status", "")
    primary_name = primary_metric.get("name", "")

    # --- metric_score: how close to target ---
    if met:
        metric_score = 1.0
    elif target > 0:
        metric_score = min(1.0, max(0.0, value / target))
    else:
        metric_score = 0.0

    # --- stability_score ---
    stability_score = 1.0
    if stability.get("nan_detected"):
        stability_score -= 0.5
    if not stability.get("loss_converged", False):
        stability_score -= 0.3
    if stability.get("plateau_detected"):
        stability_score -= 0.1
    if stability.get("overfitting_suspected"):
        stability_score -= 0.1
    stability_score = max(0.0, stability_score)
    if status in ("smoke_failed", "failed", "metrics_parse_error", "timeout"):
        stability_score = 0.0

    # --- implementation_score ---
    implementation_score = 1.0
    mech_audit = hypothesis_impl.get("mechanism_audit", {})
    metric_audit = hypothesis_impl.get("metric_audit", {})
    constraints_audit = hypothesis_impl.get("constraints_audit", {})

    if mech_audit and not mech_audit.get("implemented", True):
        implementation_score -= 0.5
    if metric_audit and not metric_audit.get("correct", True):
        implementation_score -= 0.3
    if constraints_audit and not constraints_audit.get("satisfied", True):
        implementation_score -= 0.2
    implementation_score = max(0.0, implementation_score)

    # --- trend_score: improvement across runs ---
    trend_score = 0.5  # neutral default
    if run_history and primary_name:
        prev_values = []
        for entry in run_history:
            entry_metrics = entry.get("metrics", {})
            if isinstance(entry_metrics, dict):
                prev_val = entry_metrics.get("value") if "value" in entry_metrics else entry_metrics.get(primary_name)
                if prev_val is not None:
                    prev_values.append(float(prev_val))
        if prev_values:
            latest_prev = prev_values[-1]
            if latest_prev > 0 and value > latest_prev:
                improvement_ratio = (value - latest_prev) / latest_prev
                trend_score = min(1.0, 0.5 + improvement_ratio * 2)
            elif latest_prev > 0 and value < latest_prev:
                degradation_ratio = (latest_prev - value) / latest_prev
                trend_score = max(0.0, 0.5 - degradation_ratio * 2)
    # Primary metric delta also informs trend
    primary_delta = deltas.get(primary_name)
    if primary_delta is not None and primary_delta > 0:
        trend_score = min(1.0, trend_score + 0.1)

    # --- Final weighted confidence ---
    w = CONFIDENCE_WEIGHTS
    final_confidence = round(
        metric_score * w["metric"]
        + stability_score * w["stability"]
        + implementation_score * w["implementation"]
        + trend_score * w["trend"],
        4,
    )

    # --- Explanation ---
    parts: list[str] = []
    if met:
        parts.append("target achieved")
    elif metric_score > 0.8:
        parts.append("close to target")
    elif metric_score < 0.3:
        parts.append("far from target")
    else:
        parts.append("metric below target")

    if stability_score >= 0.8:
        parts.append("training is stable")
    elif stability_score < 0.3:
        parts.append("training instability detected")

    if implementation_score >= 1.0:
        parts.append("implementation is valid")
    elif implementation_score < 0.5:
        parts.append("implementation issues detected")

    if trend_score > 0.6:
        parts.append("improving trend across runs")
    elif trend_score < 0.4:
        parts.append("degrading or stagnant trend")

    explanation = "; ".join(parts)

    return {
        "metric_score": round(metric_score, 4),
        "stability_score": round(stability_score, 4),
        "implementation_score": round(implementation_score, 4),
        "trend_score": round(trend_score, 4),
        "final_confidence": final_confidence,
        "explanation": explanation,
    }


# --- A-5b: Effect Size ---

def _compute_effect_size(
    current_value: float,
    previous_value: float,
    target: float,
) -> dict:
    """버전 간 효과 크기를 계산한다.

    single-run 비교이므로 Cohen's d 대신 상대 변화율 + 목표 대비 진전율을 사용한다.
    Returns: {relative_change, target_progress, magnitude, interpretation}
    """
    if previous_value == 0 and current_value == 0:
        return {
            "relative_change": 0.0,
            "target_progress": 0.0,
            "magnitude": "none",
            "interpretation": "no change (both zero)",
        }

    # 상대 변화율
    if previous_value != 0:
        rel_change = (current_value - previous_value) / abs(previous_value)
    else:
        rel_change = 1.0 if current_value > 0 else 0.0

    # 목표 대비 진전율: 이전→현재 변화가 이전→목표 거리의 몇 %를 해소했는지
    gap_before = target - previous_value
    gap_after = target - current_value
    if abs(gap_before) > 1e-12:
        target_progress = (gap_before - gap_after) / abs(gap_before)
    else:
        target_progress = 0.0

    # 크기 판정
    abs_rel = abs(rel_change)
    if abs_rel < EFFECT_SIZE_SMALL:
        magnitude = "negligible"
    elif abs_rel < EFFECT_SIZE_MEDIUM:
        magnitude = "small"
    elif abs_rel < EFFECT_SIZE_LARGE:
        magnitude = "medium"
    else:
        magnitude = "large"

    # 해석
    direction = "improved" if rel_change > 0 else "degraded" if rel_change < 0 else "unchanged"
    interpretation = (
        f"{direction} by {abs_rel:.2%} ({magnitude} effect). "
        f"Target gap reduced by {target_progress:.1%}."
    )

    return {
        "relative_change": round(rel_change, 6),
        "target_progress": round(target_progress, 4),
        "magnitude": magnitude,
        "interpretation": interpretation,
    }


# --- A-5c: Metric Value Validation ---

def _validate_metric_values(metrics: dict, primary_name: str) -> list[str]:
    """Metric 값이 도메인 합리 범위 내인지 검증한다.

    Returns: list of warning strings (empty if all ok).
    """
    warnings: list[str] = []
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        # NaN / Inf 검사
        if value != value:  # NaN check
            warnings.append(f"metric_nan:{key}={value}")
            continue
        if abs(value) == float("inf"):
            warnings.append(f"metric_inf:{key}={value}")
            continue

        # 범위 검사 — metric 이름의 접두어/접미어로 매칭
        matched_range = None
        key_lower = key.lower()
        for range_key, (lo, hi) in METRIC_VALID_RANGES.items():
            if key_lower == range_key or key_lower.endswith(f"_{range_key}"):
                matched_range = (range_key, lo, hi)
                break
        if matched_range:
            rk, lo, hi = matched_range
            if value < lo or value > hi:
                warnings.append(
                    f"metric_range_suspect:{key}={value} "
                    f"(expected {rk} in [{lo}, {hi}])"
                )
    return warnings


# --- A-6: Ablation Findings ---

# Patch family classification keywords
_PATCH_FAMILY_KEYWORDS: dict[str, list[str]] = {
    "augmentation":    ["augment", "transform", "flip", "crop", "noise", "jitter", "mixup", "cutout", "cutmix"],
    "capacity":        ["channel", "depth", "width", "num_layer", "n_layer", "more layer", "fewer layer", "block", "head", "hidden", "dim", "expand", "embed"],
    "optimizer":       ["optimizer", "lr", "learning_rate", "momentum", "weight_decay", "adam", "sgd", "scheduler", "warmup", "cosine"],
    "regularization":  ["dropout", "drop_path", "regulariz", "l2", "l1", "label_smooth"],
    "normalization":   ["norm", "batch_norm", "layer_norm", "group_norm", "instance_norm"],
    "loss":            ["loss", "criterion", "bce", "cross_entropy", "mse", "focal", "dice"],
}


def _classify_patch_families(patches: list[dict]) -> list[str]:
    """Classify a list of patch dicts into patch family tags."""
    families: set[str] = set()
    for patch in patches:
        text = json.dumps(patch, ensure_ascii=False).lower()
        for family, keywords in _PATCH_FAMILY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                families.add(family)
    return sorted(families)


def _infer_ablation_findings(
    pkg_dir: Path,
    primary_metric: dict,
    run_history: list[dict],
) -> list[dict]:
    """A-6: Infer micro-ablation findings from revision history and metric deltas.

    Reads previous_results.jsonl and compares patch changes vs metric changes.
    Returns list of {change_family, likely_effect, supporting_runs, confidence}.
    """
    findings: list[dict] = []
    slug = slug_from_pkg(pkg_dir)
    prev_results_path = results_dir(slug) / "previous_results.jsonl"

    if not prev_results_path.exists():
        return findings

    # Load all previous results
    prev_results: list[dict] = []
    try:
        for line in prev_results_path.read_text(encoding="utf-8").strip().split("\n"):
            line = line.strip()
            if line:
                prev_results.append(json.loads(line))
    except Exception:
        return findings

    if len(prev_results) < 2:
        return findings

    primary_name = primary_metric.get("name", "")
    if not primary_name:
        return findings

    # Compare consecutive runs to infer patch effects
    for i in range(1, len(prev_results)):
        prev_run = prev_results[i - 1]
        curr_run = prev_results[i]

        prev_primary = prev_run.get("primary_metric", {})
        curr_primary = curr_run.get("primary_metric", {})
        prev_val = float(prev_primary.get("value", 0))
        curr_val = float(curr_primary.get("value", 0))

        if prev_val == 0 and curr_val == 0:
            continue

        delta = curr_val - prev_val
        rel_delta = delta / (abs(prev_val) + 1e-12)

        # Get patch family tags from run history entries if available
        patch_families: list[str] = []
        for entry in run_history:
            if entry.get("run_id") == curr_run.get("run_id"):
                patch_families = entry.get("patch_family_tags", [])
                break

        # If no tags in history, try to infer from accepted patches
        if not patch_families:
            curr_patches = curr_run.get("accepted_patches", [])
            if curr_patches:
                patch_families = _classify_patch_families(curr_patches)

        if not patch_families:
            patch_families = ["unclassified"]

        # Only report if significant change
        if abs(rel_delta) < ABLATION_EFFECT_THRESHOLD:
            continue

        # Effect size classification
        abs_rel = abs(rel_delta)
        if abs_rel < EFFECT_SIZE_SMALL:
            effect_magnitude = "negligible"
        elif abs_rel < EFFECT_SIZE_MEDIUM:
            effect_magnitude = "small"
        elif abs_rel < EFFECT_SIZE_LARGE:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"

        n_families = len(patch_families)
        is_confounded = n_families > 1

        if delta > 0:
            effect = f"{'/'.join(patch_families)} likely beneficial (+{rel_delta:.1%}, {effect_magnitude})"
            confidence_val = min(0.8, abs_rel * 3)
        else:
            effect = f"{'/'.join(patch_families)} likely detrimental ({rel_delta:.1%}, {effect_magnitude})"
            confidence_val = min(0.7, abs_rel * 2)

        # 다중 패치 묶음이면 신뢰도 감소 + confounded 경고
        if is_confounded:
            confidence_val *= 0.6  # 다수 패치 묶음 → 신뢰도 40% 감소
            effect += f" [CONFOUNDED: {n_families} families applied together, individual effect unknown]"

        findings.append({
            "change_family": patch_families,
            "likely_effect": effect,
            "effect_magnitude": effect_magnitude,
            "confounded": is_confounded,
            "n_changes": n_families,
            "supporting_runs": [
                prev_run.get("run_id", f"v{i}"),
                curr_run.get("run_id", f"v{i + 1}"),
            ],
            "confidence": round(confidence_val, 2),
            "metric_delta": round(delta, 4),
            "relative_delta": round(rel_delta, 4),
        })

    # ── 인과 추론: 개별 패치 효과 분리 ──
    # 1단계: 단일 변경 관측에서 클린 효과 수집
    clean_effects: dict[str, list[float]] = {}  # family → [rel_delta, ...]
    for f in findings:
        if not f["confounded"] and len(f["change_family"]) == 1:
            fam = f["change_family"][0]
            clean_effects.setdefault(fam, []).append(f["relative_delta"])

    # 2단계: confounded 묶음에서 차분 귀인 시도
    for f in findings:
        if not f["confounded"]:
            continue
        families = f["change_family"]
        total_delta = f["relative_delta"]

        known_sum = 0.0
        known_families: list[str] = []
        unknown_families: list[str] = []

        for fam in families:
            if fam in clean_effects:
                avg_effect = sum(clean_effects[fam]) / len(clean_effects[fam])
                known_sum += avg_effect
                known_families.append(fam)
            else:
                unknown_families.append(fam)

        # 알려진 효과를 빼서 나머지 효과 추정
        if known_families and unknown_families:
            residual = total_delta - known_sum
            n_unknown = len(unknown_families)
            per_unknown = residual / n_unknown if n_unknown else 0

            f["causal_attribution"] = {
                "method": "differential",
                "known_effects": {fam: round(sum(clean_effects[fam]) / len(clean_effects[fam]), 4)
                                  for fam in known_families},
                "residual_effect": round(residual, 4),
                "estimated_per_unknown": round(per_unknown, 4),
                "unknown_families": unknown_families,
                "attribution_confidence": round(
                    min(0.6, 0.3 + 0.1 * len(known_families)), 2
                ),
            }
            f["likely_effect"] += (
                f" [DIFFERENTIAL: known={known_families} ({known_sum:+.1%}), "
                f"residual for {unknown_families}={residual:+.1%}]"
            )
            # 차분 귀인이 가능하면 신뢰도 일부 회복
            f["confidence"] = round(min(f["confidence"] * 1.3, 0.7), 2)
        elif not unknown_families and known_families:
            # 모든 family의 클린 효과를 알고 있음 → 합산 vs 실제 비교
            synergy = total_delta - known_sum
            f["causal_attribution"] = {
                "method": "additive_check",
                "known_effects": {fam: round(sum(clean_effects[fam]) / len(clean_effects[fam]), 4)
                                  for fam in known_families},
                "expected_sum": round(known_sum, 4),
                "actual_total": round(total_delta, 4),
                "synergy_or_interference": round(synergy, 4),
                "attribution_confidence": 0.5,
            }
            if abs(synergy) > ABLATION_EFFECT_THRESHOLD:
                f["likely_effect"] += (
                    f" [SYNERGY: expected sum={known_sum:+.1%}, "
                    f"actual={total_delta:+.1%}, interaction={synergy:+.1%}]"
                )

    # 3단계: 클린 효과 요약을 findings에 추가
    if clean_effects:
        summary_lines = []
        for fam, effects in sorted(clean_effects.items()):
            avg = sum(effects) / len(effects)
            n = len(effects)
            summary_lines.append(f"{fam}: avg={avg:+.1%} (n={n})")
        findings.append({
            "change_family": ["_summary"],
            "likely_effect": "Clean single-change effects: " + ", ".join(summary_lines),
            "effect_magnitude": "summary",
            "confounded": False,
            "n_changes": 0,
            "supporting_runs": [],
            "confidence": 0.0,
            "metric_delta": 0.0,
            "relative_delta": 0.0,
            "causal_attribution": {
                "method": "clean_observation_summary",
                "family_effects": {fam: round(sum(effs) / len(effs), 4)
                                   for fam, effs in clean_effects.items()},
            },
        })

    return findings


# ──────────────────────────────────────────────────────────
# A-7: Senior Researcher Diagnostics
#   a) 판세를 바꾸는 핵심 근거 압축
#   b) 희소 증거 기반 과학적 베팅
#   c) 문제 재정의 신호 감지
# ──────────────────────────────────────────────────────────

def _triage_pivotal_evidence(
    ablation_findings: list[dict],
    bottleneck_candidates: list[dict],
    effect_size: dict,
    primary_metric: dict,
) -> list[dict]:
    """A-7a: 모든 증거 중 '판세를 바꾸는' 핵심만 추출한다 (최대 3개).

    10년차 박사는 20개 근거 중 진짜 중요한 2-3개만 골라 판단한다.
    """
    pivots: list[dict] = []
    target = float(primary_metric.get("target", 0))
    value = float(primary_metric.get("value", 0))
    gap = target - value

    # 1. ablation에서 가장 큰 효과
    for f in ablation_findings:
        if f.get("change_family") == ["_summary"]:
            continue
        rel = abs(f.get("relative_delta", 0))
        if rel < EFFECT_SIZE_MEDIUM:
            continue
        direction = "beneficial" if f.get("relative_delta", 0) > 0 else "detrimental"
        gap_coverage = f"{rel / (abs(gap) + 1e-12) * 100:.0f}% of gap" if gap > 0 else ""
        pivots.append({
            "finding": f"{f['change_family']}: {direction} ({f.get('relative_delta',0):+.1%})",
            "impact": f.get("effect_magnitude", ""),
            "why_pivotal": gap_coverage or f"Largest effect ({f.get('effect_magnitude','')})",
            "action": "repeat" if direction == "beneficial" else "revert",
        })

    # 2. critical bottleneck
    for bn in bottleneck_candidates:
        if bn.get("severity") in ("critical", "high"):
            pivots.append({
                "finding": f"Bottleneck: {bn['name']}",
                "impact": bn["severity"],
                "why_pivotal": "; ".join(bn.get("evidence", [])[:2]),
                "action": "resolve_first",
            })

    # 3. effect size가 큰 경우 — 방향 검증
    if effect_size.get("magnitude") in ("medium", "large"):
        pivots.append({
            "finding": f"Latest: {effect_size.get('magnitude')} effect ({effect_size.get('relative_change',0):+.1%})",
            "impact": effect_size["magnitude"],
            "why_pivotal": "Direction validated" if effect_size.get("relative_change", 0) > 0 else "Regression detected",
            "action": "continue" if effect_size.get("relative_change", 0) > 0 else "investigate",
        })

    pivots.sort(key=lambda p: {"critical": 4, "high": 3, "large": 3, "medium": 2}.get(p["impact"], 0), reverse=True)
    return pivots[:3]


def _frame_scientific_bet(
    primary_metric: dict,
    confidence_model: dict,
    run_history: list[dict],
    bottleneck_candidates: list[dict],
    hypothesis_impl: dict,
) -> dict:
    """A-7b: 희소한 증거에서 과학적 베팅을 구조화한다.

    "X에 베팅하며, Y가 나오면 수정한다" 형식의 calibrated judgment.
    bet_grade: A(강한 베팅) ~ D(보류)
    """
    conf = confidence_model.get("final_confidence", 0)
    met = primary_metric.get("met", False)
    n_runs = len(run_history)
    value = float(primary_metric.get("value", 0))
    target = float(primary_metric.get("target", 0))
    attainment = value / target if target > 0 else 0

    # 지지/반대 증거
    evidence_for = [k for k, v in {
        "metric_close": confidence_model.get("metric_score", 0) > 0.7,
        "stable_training": confidence_model.get("stability_score", 0) > 0.7,
        "mechanism_ok": confidence_model.get("implementation_score", 0) >= 1.0,
        "improving_trend": confidence_model.get("trend_score", 0) > 0.6,
    }.items() if v]

    evidence_against = [
        f"{bn['name']} ({bn.get('severity', '')})"
        for bn in bottleneck_candidates if bn.get("severity") in ("critical", "high")
    ]
    mech = hypothesis_impl.get("mechanism_audit", {})
    if mech and not mech.get("implemented", True):
        evidence_against.append("mechanism_not_implemented")

    # 등급 결정
    if conf >= 0.8 and not evidence_against:
        grade, hedge = "A", "Proceed. Validate on held-out data."
    elif conf >= 0.6 and len(evidence_against) <= 1:
        grade, hedge = "B", f"Proceed, monitor: {evidence_against[0] if evidence_against else 'trend'}."
    elif conf >= 0.4:
        grade, hedge = "C", f"Conditional. Resolve first: {'; '.join(evidence_against[:2])}."
    else:
        grade, hedge = "D", "Insufficient evidence. Pivot or collect more data."

    if n_runs <= 1:
        grade = max(grade, "C")
        hedge += " [SPARSE: single run, wide confidence bounds]"

    return {
        "claim": f"{'Target met' if met else f'Attainment {attainment:.0%}'}: {primary_metric.get('name','')}={value:.4f}",
        "bet_grade": grade,
        "confidence": round(conf, 3),
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
        "hedge": hedge,
        "n_runs": n_runs,
    }


def _detect_problem_reframe(
    run_history: list[dict],
    bottleneck_candidates: list[dict],
    stability: dict,
    primary_metric: dict,
    hypothesis_impl: dict,
) -> dict:
    """A-7c: 문제 자체를 다시 정의해야 하는 신호를 감지한다.

    '해법이 틀린 것'이 아니라 '문제가 잘못 정의된 것'일 때의 패턴:
    - 모든 변경이 효과 없음 → 데이터/태스크 문제
    - mechanism 구현 + 반복 실패 → 가설 전제 오류
    - 학습 안정적 + metric 낮음 → 잘못된 신호 학습
    """
    signals: list[str] = []
    n_runs = len(run_history)

    if primary_metric.get("met", False) or n_runs < 2:
        return {"reframe_detected": False, "signals": [], "suggested_reframe": ""}

    primary_name = primary_metric.get("name", "")
    target = float(primary_metric.get("target", 0))
    value = float(primary_metric.get("value", 0))

    # 신호 1: 3+ 실행에서 metric 정체 (전체 변동 < 5%)
    if n_runs >= 3:
        vals = []
        for e in run_history:
            m = e.get("metrics", {})
            v = m.get("value") if "value" in m else m.get(primary_name)
            if v is not None:
                vals.append(float(v))
        if len(vals) >= 3:
            rng = max(vals) - min(vals)
            mean = sum(vals) / len(vals)
            if mean > 0 and rng / mean < 0.05:
                signals.append(f"plateau_across_{len(vals)}_runs ({rng/mean:.1%} variation)")

    # 신호 2: mechanism 구현됨 + 반복 실패 → 전제 문제
    mech = hypothesis_impl.get("mechanism_audit", {})
    if mech.get("implemented", False) and n_runs >= 3:
        fails = sum(1 for e in run_history if e.get("metrics", {}).get("met") is False)
        if fails >= 3:
            signals.append(f"mechanism_ok_but_{fails}_failures — premise may be wrong")

    # 신호 3: 다수 bottleneck 동시 존재 → 문제 분해 필요
    bn_names = set(bn.get("name", "") for bn in bottleneck_candidates)
    if len(bn_names) >= 3:
        signals.append(f"{len(bn_names)}_bottleneck_types — problem needs decomposition")

    # 신호 4: 안정적 학습 + 낮은 metric → 잘못된 신호 학습
    if (stability.get("loss_converged") and not stability.get("nan_detected")
            and not stability.get("overfitting_suspected")):
        attainment = value / target if target > 0 else 0
        if attainment < 0.7:
            signals.append(f"stable_converged_but_{attainment:.0%}_attainment — wrong signal?")

    reframe = len(signals) >= 2
    suggestion = ""
    if reframe:
        if any("stable" in s or "wrong" in s for s in signals):
            suggestion = "Audit data quality/labeling/task definition before further model iteration."
        elif any("premise" in s for s in signals):
            suggestion = "Re-examine hypothesis premise. The mechanism may address the wrong bottleneck."
        else:
            suggestion = "Decompose the problem. Address the most constrained sub-problem first."

    return {
        "reframe_detected": reframe,
        "signals": signals,
        "suggested_reframe": suggestion,
        "confidence": round(min(0.8, len(signals) * 0.25), 2),
    }


# ──────────────────────────────────────────────────────────
# result_summary.json 생성 (A-1: value collector only)
# ──────────────────────────────────────────────────────────

def _build_result_summary(
    pkg_dir: Path,
    run_result: dict,
    spec: dict,
    previous_run_id: str | None,
    previous_metrics: dict,
    run_history: list[dict] | None = None,
) -> dict:
    """Build canonical result_summary.json (A-1: value collector only).

    All diagnostic logic is delegated to helper functions:
        _infer_training_stability   (A-2)
        _infer_bottlenecks          (A-3)
        _infer_recommended_actions  (A-4)
        _estimate_confidence        (A-5)
        _infer_ablation_findings    (A-6)
    """
    if run_history is None:
        run_history = []

    primary_name  = spec["evaluation_config"]["primary_metric"]
    target_value  = spec["evaluation_config"]["target_value"]
    primary_value = float(run_result["metrics"].get(primary_name, 0.0))
    primary_met   = primary_value >= target_value

    secondary = [
        {"name": k, "value": float(v), "unit": ""}
        for k, v in run_result["metrics"].items()
        if k != primary_name
    ]

    deltas = {
        k: round(float(v) - float(previous_metrics[k]), 4)
        for k, v in run_result["metrics"].items()
        if k in previous_metrics
    }

    run_id = (
        f"{spec['topic_slug']}_v{spec['experiment_version']}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    primary_metric_obj = {
        "name":   primary_name,
        "value":  primary_value,
        "unit":   spec.get("evaluation_config", {}).get("metric_units", {}).get(primary_name, ""),
        "target": float(target_value),
        "met":    primary_met,
    }

    # runner metadata (runner abstraction에서 제공)
    runner_meta = run_result.get("metadata", {})

    # hypothesis implementation audit
    hypothesis_impl = _load_hypothesis_audit(pkg_dir)

    # --- A-2: Training stability ---
    stability = _infer_training_stability(run_result, run_result.get("metrics", {}))

    # --- A-3: Bottleneck candidates ---
    bottleneck_candidates = _infer_bottlenecks(
        stability, primary_metric_obj, hypothesis_impl, run_result,
    )

    # --- A-4: Recommended next actions ---
    recommended_actions = _infer_recommended_actions(
        stability, primary_metric_obj, hypothesis_impl,
        bottleneck_candidates, run_history, run_result,
    )

    # --- A-5: Confidence model ---
    confidence_model = _estimate_confidence(
        primary_metric_obj, stability, hypothesis_impl,
        deltas, run_history, run_result,
    )

    # --- A-5b: Effect size (vs previous version) ---
    prev_primary_value = float(previous_metrics.get(primary_name, 0))
    effect_size = _compute_effect_size(primary_value, prev_primary_value, target_value)

    # --- A-5c: Metric value validation ---
    metric_warnings = _validate_metric_values(run_result.get("metrics", {}), primary_name)
    if metric_warnings:
        print(f"    [METRIC 경고] {', '.join(metric_warnings)}")

    # --- A-6: Ablation findings ---
    ablation_findings = _infer_ablation_findings(
        pkg_dir, primary_metric_obj, run_history,
    )

    summary = {
        "schema_version":   "1.0",
        "run_id":           run_id,
        "spec_id":          spec.get("spec_id", ""),
        "hypothesis_id":    spec.get("hypothesis_id", ""),
        "experiment_version": spec.get("experiment_version", 1),
        "status":           run_result["status"],
        "primary_metric":   primary_metric_obj,
        "all_metrics":        run_result.get("metrics", {}),
        "secondary_metrics":  secondary,
        "deltas_vs_baseline": {
            "baseline_run_id": previous_run_id,
            "deltas":          deltas,
        },
        # A-2: real training stability diagnostics
        "training_stability": stability,
        # A-3: deterministic bottleneck candidates
        "bottleneck_candidates": bottleneck_candidates,
        # A-6: ablation findings from revision history
        "ablation_findings": ablation_findings,
        # A-5: backward-compatible float confidence
        "confidence": confidence_model["final_confidence"],
        # A-5: full confidence model with sub-scores
        "confidence_model": confidence_model,
        # A-5b: effect size vs previous version
        "effect_size": effect_size,
        # A-5c: metric value validation warnings
        "metric_warnings": metric_warnings,
        # A-4: deterministic recommended next actions (prior for LLM decision)
        "recommended_next_actions": recommended_actions,
        # A-7: senior researcher diagnostics
        "pivotal_evidence": _triage_pivotal_evidence(
            ablation_findings, bottleneck_candidates, effect_size, primary_metric_obj),
        "scientific_bet": _frame_scientific_bet(
            primary_metric_obj, confidence_model, run_history or [],
            bottleneck_candidates, hypothesis_impl),
        "problem_reframe": _detect_problem_reframe(
            run_history or [], bottleneck_candidates, stability,
            primary_metric_obj, hypothesis_impl),
        "stderr_tail": run_result.get("stderr_tail", [])[-50:],
        "stdout_tail": (
            run_result.get("stdout_lines", [])[-50:]
            if run_result["status"] == "metrics_parse_error" else []
        ),
        # runner metadata: sanitized (토큰/키 제거 후 보존)
        "runner_metadata": runner_meta,  # _make_result에서 이미 sanitize됨
        # hypothesis implementation audit 결과 로드
        "hypothesis_implementation": hypothesis_impl,
        "created_at": datetime.now().isoformat(),
    }

    # experiments/{slug}/results/vN/result_summary.json
    slug = slug_from_pkg(pkg_dir)
    ver = version_from_pkg(pkg_dir)
    ver_results_dir = result_version_dir(slug, ver)
    ver_results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ver_results_dir / "result_summary.json"
    clean_summary = _sanitize_summary(summary)
    summary_path.write_text(json.dumps(clean_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # runner_metadata.json 별도 저장 (sanitized)
    (ver_results_dir / "runner_metadata.json").write_text(
        json.dumps(clean_summary.get("runner_metadata", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"    [저장] {summary_path} (sanitized)")
    return summary


# ──────────────────────────────────────────────────────────
# 결과 분석 — Multi-model Proposal-Review-Merge
# ──────────────────────────────────────────────────────────

def _gpt_interpret_results(
    summary: dict,
    hypothesis: dict,
    spec: dict,
    run_history: list[dict],
) -> dict:
    """GPT가 실험 결과를 심층 해석하고 revision 방향을 제안한다 (주요 해석자).

    해석에 집중하며 코드 패치는 생성하지 않는다.
    출력: suggested_path, evidence_strength, root_cause_analysis,
           hypothesis_validity_assessment, bottleneck_candidates,
           confidence, why_not_other_paths
    """
    primary = summary["primary_metric"]
    hyp     = hypothesis.get("hypothesis", {})
    ev_cfg  = spec.get("evaluation_config", {})

    system_msg = (
        "You are a deep learning research analyst (GPT role). "
        "Your job is to perform DEEP INTERPRETATION of experiment results. "
        "Do NOT generate code or propose patches here — that is a separate step. "
        "Focus on: root cause analysis, hypothesis validity, and revision direction. "
        "Return valid JSON only."
    )

    user_msg = f"""Interpret the experiment results and suggest a revision direction.

## Hypothesis
{hyp.get('statement_kr', hyp.get('statement', ''))}
Key mechanism: {hyp.get('expected_mechanism', hyp.get('key_mechanism', hyp.get('mechanism', '')))}

## Experiment Spec
- primary_metric: {ev_cfg.get('primary_metric', primary['name'])} ≥ {ev_cfg.get('target_value', primary['target'])}
- secondary_metrics: {ev_cfg.get('secondary_metrics', [])}

## Run History ({len(run_history)} runs)
{json.dumps(run_history, ensure_ascii=False, indent=2)}

## Latest Result
- status: {summary['status']}
- {primary['name']}: {primary['value']:.4f} (target={primary['target']:.2f}, met={primary['met']})
- training_stability: {json.dumps(summary.get('training_stability', {}), ensure_ascii=False)}
- deltas_vs_baseline: {json.dumps(summary.get('deltas_vs_baseline', {}), ensure_ascii=False)}

## Hypothesis Implementation Status
{json.dumps(summary.get('hypothesis_implementation', {}), ensure_ascii=False, indent=2)}

## Deterministic Diagnostic Prior (summary layer)
- training_stability: {json.dumps(summary.get('training_stability', {}), ensure_ascii=False)}
- bottleneck_candidates: {json.dumps(summary.get('bottleneck_candidates', []), ensure_ascii=False)}
- recommended_next_actions (rule-based prior): {json.dumps(summary.get('recommended_next_actions', []), ensure_ascii=False)}
- confidence_model: {json.dumps(summary.get('confidence_model', {}), ensure_ascii=False)}

## Senior Researcher Diagnostics (A-7)
- pivotal_evidence (판세를 바꾸는 핵심): {json.dumps(summary.get('pivotal_evidence', []), ensure_ascii=False)}
- scientific_bet (과학적 베팅): {json.dumps(summary.get('scientific_bet', {}), ensure_ascii=False)}
- problem_reframe (문제 재정의 신호): {json.dumps(summary.get('problem_reframe', {}), ensure_ascii=False)}

NOTE: The above diagnostic prior is deterministic and rule-based.
You may agree or disagree, but must justify departures from this prior.
Pay special attention to problem_reframe signals — if reframe_detected=true,
consider whether the problem definition itself needs revision, not just the solution.

## Revision criteria (strict)
- Path A: implementation issue — hypothesis still valid (code bug, config, training instability)
  - 특히 mechanism이 코드에 구현되지 않은 경우 (mechanism_audit.implemented=false) → Path A 우선
- Path B: hypothesis directionally valid but needs refinement (50-80% target met, scope too broad)
- Path C: hypothesis CORE MECHANISM contradicted by repeated strong evidence (requires ≥3 runs)
  - mechanism 미구현 상태에서는 Path C 절대 불가 (구현 후 재평가 필요)
- done: target metric achieved

Rules: single failure cannot go to Path C. Escalation must follow A → B → C.

IMPORTANT DISTINCTION:
- "mechanism not implemented" (코드에 mechanism 미구현) → 반드시 Path A (구현 문제)
- "mechanism implemented but unsupported by evidence" (구현했으나 실험적 반박) → Path B/C 검토 가능

Return JSON only:
{{
  "suggested_path": "A|B|C|done",
  "evidence_strength": "low|medium|high",
  "confidence": "low|medium|high",
  "root_cause_analysis": "...",
  "hypothesis_validity_assessment": "...",
  "bottleneck_candidates": ["..."],
  "improvement_suggestions": ["..."],
  "why_not_other_paths": "..."
}}"""

    p_hash = prompt_hash(user_msg)
    print(f"  [해석 / GPT]    결과 심층 분석... (prompt_hash={p_hash})")
    try:
        client = get_openai_client()
        resp = llm_retry(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            label="GPT interpret",
        )
        result = json.loads(resp.choices[0].message.content)
        result["_prompt_hash"] = p_hash
        print(f"    → suggested_path={result.get('suggested_path')}, "
              f"evidence={result.get('evidence_strength')}, "
              f"confidence={result.get('confidence')}")
        return result
    except Exception as e:
        print(f"    [경고] GPT 결과 해석 실패: {e}")
        return {
            "suggested_path": "A",
            "evidence_strength": "low",
            "confidence": "low",
            "root_cause_analysis": f"GPT call failed: {e}",
            "hypothesis_validity_assessment": "unknown",
            "bottleneck_candidates": [],
            "improvement_suggestions": [],
            "why_not_other_paths": "",
        }


def _gemini_short_diagnosis(
    summary: dict,
    hypothesis: dict,
    gpt_interpretation: dict,
) -> dict:
    """Gemini가 **독립** 진단을 수행한다.

    GPT 해석을 사전에 보지 않고(blind) 동일한 데이터만으로 독립 판단한다.
    이후 합의 레이어에서 GPT 결과와 비교하여 agreement 수준을 결정론적으로 판정.
    코드를 작성하거나 패치를 제안하지 않는다.
    출력: suggested_path, short_diagnosis, root_cause, main_risk, confidence
    """
    primary = summary["primary_metric"]
    hyp     = hypothesis.get("hypothesis", {})

    # ⚠️ GPT 해석을 Gemini에 전달하지 않음 — anchoring bias 방지
    prompt = f"""You are a deep learning research diagnostician (Gemini role).
Provide a SHORT **independent** diagnosis based ONLY on the data below.
Do NOT write code. Do NOT propose code changes. Return valid JSON only.

## Hypothesis
{hyp.get('statement_kr', hyp.get('statement', ''))}

## Latest Result
- status: {summary['status']}
- {primary['name']}: {primary['value']:.4f} (target={primary['target']:.2f}, met={primary['met']})
- stability: {json.dumps(summary.get('training_stability', {}), ensure_ascii=False)}

## Deterministic Diagnostic Prior (summary layer — rule-based, for reference)
- bottleneck_candidates: {json.dumps(summary.get('bottleneck_candidates', []), ensure_ascii=False)}
- recommended_next_actions (rule-based): {json.dumps(summary.get('recommended_next_actions', []), ensure_ascii=False)}
- confidence_model: {json.dumps(summary.get('confidence_model', {}), ensure_ascii=False)}

## Hypothesis Implementation Status
{json.dumps(summary.get('hypothesis_implementation', {}), ensure_ascii=False, indent=2)}

## Effect Size (vs previous version)
{json.dumps(summary.get('effect_size', {}), ensure_ascii=False)}

## Revision criteria (strict)
- Path A: implementation issue — hypothesis still valid
- Path B: hypothesis directionally valid but needs refinement (50-80% target met)
- Path C: hypothesis CORE MECHANISM contradicted by repeated strong evidence (requires >=3 runs)
- done: target metric achieved

Your independent diagnosis:
{{
  "suggested_path": "A|B|C|done",
  "short_diagnosis": "one clear sentence explaining the main issue",
  "root_cause": "suspected root cause of current performance gap",
  "main_risk": "biggest risk if current approach continues unchanged",
  "confidence": "low|medium|high"
}}"""

    p_hash = prompt_hash(prompt)
    print(f"  [진단 / Gemini] 독립 진단... (prompt_hash={p_hash})")
    try:
        model  = get_gemini_model()
        resp   = llm_retry(model.generate_content, prompt, label="Gemini diagnose")
        text   = resp.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        result = json.loads(text)

        # 결정론적 agreement 판정 (GPT 결과와 비교)
        gpt_path = gpt_interpretation.get("suggested_path", "A")
        gem_path = result.get("suggested_path", "A")
        if gem_path == gpt_path:
            agreement = "agree"
            disagreement_reason = ""
        elif {gem_path, gpt_path} <= {"A", "B"} or {gem_path, gpt_path} <= {"B", "C"}:
            agreement = "partial"
            disagreement_reason = (
                f"Gemini suggests {gem_path} vs GPT {gpt_path}. "
                f"Gemini root cause: {result.get('root_cause', '')}"
            )
        else:
            agreement = "disagree"
            disagreement_reason = (
                f"Gemini suggests {gem_path} vs GPT {gpt_path} (2+ levels apart). "
                f"Gemini root cause: {result.get('root_cause', '')}"
            )
        result["agreement_with_gpt"] = agreement
        result["disagreement_reason"] = disagreement_reason
        result["_prompt_hash"] = p_hash

        print(f"    → suggested_path={gem_path}, "
              f"agreement={agreement} (blind comparison with GPT={gpt_path})")
        return result
    except Exception as e:
        print(f"    [경고] Gemini 2차 진단 실패: {e}")
        return {
            "suggested_path": gpt_interpretation.get("suggested_path", "A"),
            "short_diagnosis": f"Gemini call failed: {e}",
            "agreement_with_gpt": "agree",
            "disagreement_reason": "",
            "root_cause": "",
            "main_risk": "",
            "confidence": "low",
        }


def _build_consensus(
    gpt_interpretation: dict,
    gemini_diagnosis: dict,
    run_history: list[dict],
    primary: dict,
) -> dict:
    """GPT 해석 + Gemini 진단을 통합하여 합의 요약을 생성한다.

    Decision policy:
    - done: target met → 즉시 허용
    - Path A: 약한 합의도 허용 (구현 개선)
    - Path B: 중간 합의 필요 (가설 정제)
    - Path C: 강한 합의 + ≥3회 실행 + high evidence 필요 (가설 교체)
    """
    gpt_path  = gpt_interpretation.get("suggested_path", "A")
    gem_path  = gemini_diagnosis.get("suggested_path", "A")
    gpt_ev    = gpt_interpretation.get("evidence_strength", "low")
    gpt_conf  = gpt_interpretation.get("confidence", "medium")
    gem_conf  = gemini_diagnosis.get("confidence", "medium")
    gem_agree = gemini_diagnosis.get("agreement_with_gpt", "agree")
    n_runs    = len(run_history)

    # confidence 수치 변환
    gpt_conf_val = CONFIDENCE_LEVEL_MAP.get(gpt_conf, 0.5)
    gem_conf_val = CONFIDENCE_LEVEL_MAP.get(gem_conf, 0.5)

    # path 거리 (동일=0, 인접=0.5, 2단계=1.0)
    path_dist = PATH_DISTANCE.get((gpt_path, gem_path), 0.75)

    # consensus_strength: confidence 가중 합의 점수
    # = 평균 confidence × (1 - path 거리)
    # 동일 path + high confidence → ~0.9, 다른 path + low confidence → ~0.09
    consensus_strength = round(
        (gpt_conf_val + gem_conf_val) / 2 * (1 - path_dist), 4
    )

    # 합의 수준 계산 (기존 호환 유지 + consensus_strength 반영)
    if gpt_path == gem_path and gem_agree == "agree":
        agreement_level = "strong"
    elif gpt_path == gem_path or gem_agree in ("agree", "partial"):
        agreement_level = "medium"
    else:
        agreement_level = "weak"

    # Path B 명시적 허용 조건 (모두 충족해야 B 허용)
    # 1. GPT가 B 또는 C를 제안해야 함
    # 2. Gemini가 강한 반대(disagree)가 아니어야 함
    # 3. consensus_strength ≥ 임계값 (confidence 가중)
    # 4. 실행 횟수 ≥ 2
    def _path_b_allowed() -> bool:
        return (
            gpt_path in ("B", "C")
            and gem_agree != "disagree"
            and consensus_strength >= CONSENSUS_STRENGTH_PATH_B
            and n_runs >= 2
        )

    # 합의 경로 후보 — 과도한 에스컬레이션 방지
    if primary["met"]:
        consensus_path = "done"
    elif gpt_path == "C" and (n_runs < 3 or gpt_ev != "high" or agreement_level != "strong"
                              or consensus_strength < CONSENSUS_STRENGTH_PATH_C):
        # Path C 요건 미충족 → B 시도, B 요건도 미충족 시 A로 하향
        consensus_path = "B" if _path_b_allowed() else "A"
    elif gpt_path == "B" and not _path_b_allowed():
        consensus_path = "A"   # Path B 게이팅 미충족 → A로 하향
    else:
        consensus_path = gpt_path

    # 에스컬레이션 위험
    escalation_risk = "low"
    if consensus_path == "C":
        escalation_risk = "blocked" if (n_runs < 3 or agreement_level != "strong") else "high"
    elif consensus_path == "B":
        escalation_risk = "medium"

    major_disagreements: list[str] = []
    if gpt_path != gem_path:
        major_disagreements.append(f"GPT suggests {gpt_path}, Gemini suggests {gem_path}")
    if gem_agree == "disagree":
        major_disagreements.append(
            f"Gemini disagrees: {gemini_diagnosis.get('disagreement_reason', '')}"
        )

    notes: list[str] = []
    if escalation_risk == "blocked":
        notes.append(
            f"Path C blocked (needs ≥3 runs and strong evidence; current n_runs={n_runs}, "
            f"evidence={gpt_ev}, agreement={agreement_level}) — consider B"
        )
    # Path B 게이팅 미충족 시 사유 명시
    if gpt_path in ("B", "C") and not _path_b_allowed():
        reasons: list[str] = []
        if gpt_path not in ("B", "C"):
            reasons.append("GPT did not suggest B or C")
        if gem_agree == "disagree":
            reasons.append("Gemini strongly disagrees")
        if agreement_level == "weak":
            reasons.append(f"agreement too weak ({agreement_level})")
        if n_runs < 2:
            reasons.append(f"insufficient runs (n_runs={n_runs}, need ≥2)")
        notes.append(f"Path B gating not met: {'; '.join(reasons)} — downgraded to A")
    if gemini_diagnosis.get("main_risk"):
        notes.append(f"Risk: {gemini_diagnosis['main_risk']}")

    return {
        "consensus_path_candidate": consensus_path,
        "agreement_level":          agreement_level,
        "consensus_strength":       consensus_strength,
        "gpt_confidence":           gpt_conf,
        "gemini_confidence":        gem_conf,
        "major_disagreements":      major_disagreements,
        "escalation_risk":          escalation_risk,
        "notes_for_claude":         " | ".join(notes) if notes else "none",
        "gpt_suggested":            gpt_path,
        "gemini_suggested":         gem_path,
        "n_runs":                   n_runs,
    }


def _gpt_propose_improvements(
    pkg_dir: Path,
    summary: dict,
    gpt_interpretation: dict,
) -> dict:
    """GPT/Codex가 Path A 시에만 코드 개선 패치를 제안한다 (patch-only 역할).

    해석이나 path 결정 권한 없음. 오직 구현 레벨 패치만 제안.
    GPT 해석 결과를 컨텍스트로 활용하여 더 정확한 패치를 생성한다.
    출력: patches (spec_field, breaks_comparability 포함), expected_improvement, confidence
    """
    primary = summary["primary_metric"]

    model_code   = (pkg_dir / "model.py").read_text(encoding="utf-8")   if (pkg_dir / "model.py").exists()   else ""
    default_yaml = (pkg_dir / "configs/default.yaml").read_text(encoding="utf-8") if (pkg_dir / "configs/default.yaml").exists() else ""

    system_msg = (
        "You are a PyTorch optimization engineer (GPT/Codex patch role). "
        "Propose MINIMAL, targeted code patches to improve experiment performance. "
        "Rules: NO full-file rewrites. NO hypothesis revision. NO path decisions (A/B/C). "
        "Patches must be implementation-level only (code, config, training setup). "
        "Each patch must be spec-linked and must NOT break result comparability unless absolutely necessary. "
        "Return valid JSON only."
    )

    user_msg = f"""Propose targeted implementation patches for Path A improvement.

## Current Result
- {primary['name']}: {primary['value']:.4f} → target: {primary['target']:.2f}
- gap: {primary['target'] - primary['value']:.4f}
- status: {summary['status']}

## GPT Interpretation (root cause context — do not reproduce, use as guidance)
- root_cause: {gpt_interpretation.get('root_cause_analysis', '')}
- bottlenecks: {gpt_interpretation.get('bottleneck_candidates', [])}
- improvement_suggestions: {gpt_interpretation.get('improvement_suggestions', [])}

## Current model.py (excerpt)
```python
{model_code[:2500]}
```

## Current default.yaml
```yaml
{default_yaml[:1000]}
```

Propose minimal patches. Each patch must:
- target a specific file (not full rewrite)
- be linked to a spec field (e.g. model_architecture, training_config, evaluation_config)
- state whether it breaks result comparability with previous runs

Return JSON only:
{{
  "patches": [
    {{
      "target_file": "model.py|module.py|configs/default.yaml",
      "spec_field": "model_architecture|training_config|evaluation_config",
      "rationale": "...",
      "hypothesis_alignment_check": "...",
      "breaks_comparability": false,
      "complexity_delta_loc": 5,
      "changes": [
        {{"type": "replace", "old": "exact snippet", "new": "improved snippet"}}
      ]
    }}
  ],
  "expected_improvement": "...",
  "confidence": "low|medium|high"
}}"""

    p_hash = prompt_hash(user_msg)
    print(f"  [패치 / GPT]    Path A 구현 패치 제안... (prompt_hash={p_hash})")
    try:
        client = get_openai_client()
        resp = llm_retry(
            client.chat.completions.create,
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            label="GPT patch",
        )
        result = json.loads(resp.choices[0].message.content)
        result["_prompt_hash"] = p_hash
        print(f"    → {len(result.get('patches', []))}개 패치 제안 "
              f"(confidence={result.get('confidence')})")
        return result
    except Exception as e:
        print(f"    [경고] GPT 패치 제안 실패: {e}")
        return {"patches": [], "expected_improvement": f"GPT call failed: {e}", "confidence": "low"}


def _claude_final_decision(
    pkg_dir: Path,
    summary: dict,
    hypothesis: dict,
    run_history: list[dict],
    gpt_interpretation: dict,
    gemini_diagnosis: dict,
    consensus: dict,
    gpt_patches: dict,
    max_rounds: int,
    stage7_ballot_decisions: list[dict] | None = None,
) -> dict:
    """Claude가 override reviewer로서 합의 경로를 검토한다 (consensus-locked).

    Claude는 자유 선택자가 아니라 합의 결과의 override reviewer이다.
    consensus_path_candidate가 기본 path이며, override는 escalation만 허용된다.

    Task 6: consensus_path_candidate를 기본 path로 잠금
    Task 7: override는 escalation(A->B, B->C)만 허용
    Task 8: accepted_patch_indexes는 Stage 7 ballot 결과를 재사용

    출력: structured decision payload (Task 10)
    """
    primary  = summary["primary_metric"]
    n_runs   = len(run_history)
    status   = summary["status"]
    patches  = gpt_patches.get("patches", [])

    candidate_path = consensus.get("consensus_path_candidate", "A")
    agreement_lvl  = consensus.get("agreement_level", "weak")
    gpt_path       = gpt_interpretation.get("suggested_path", "A")
    gem_agree      = gemini_diagnosis.get("agreement_with_gpt", "agree")
    gpt_ev         = gpt_interpretation.get("evidence_strength", "low")

    # Stage 7 ballot 기반 accepted_patch_indexes (Task 8)
    ballot_accepted_indexes: list[int] = []
    accepted_patch_source = "stage7_ballot"
    if stage7_ballot_decisions:
        ballot_accepted_indexes = [
            d["patch_index"] for d in stage7_ballot_decisions
            if d.get("final_decision") in ("accept", "ambiguous")
        ]
    else:
        # ballot 없으면 전체 패치 (하위 호환)
        ballot_accepted_indexes = list(range(len(patches)))
        accepted_patch_source = "no_ballot_fallback"

    # ── 빠른 결정 — 목표 달성 ────────────────────────────────
    if primary["met"]:
        return {
            "path": "done",
            "candidate_path": "done",
            "candidate_source": "consensus_layer",
            "claude_override": False,
            "override_from": None,
            "override_to": None,
            "override_rule_check": {},
            "final_path": "done",
            "decision_reason": f"{primary['name']}={primary['value']:.4f} >= {primary['target']:.2f}",
            "justification":   f"{primary['name']}={primary['value']:.4f} >= {primary['target']:.2f}",
            "consensus_level": "strong",
            "improvement_hints": "",
            "evidence_strength": "high",
            "accepted_patch_indexes": [],
            "rejected_patch_indexes": [],
            "accepted_gpt_patches":   [],
            "accepted_patch_source": accepted_patch_source,
        }

    # ── 빠른 결정 — 실행 에러 (Path A 기본) ──────────────────
    if status in ("smoke_failed", "failed", "metrics_parse_error", "timeout"):
        accepted = ballot_accepted_indexes if ballot_accepted_indexes else list(range(len(patches)))
        hints = (
            f"status={status}. GPT root_cause: {gpt_interpretation.get('root_cause_analysis', '')}. "
            f"Applying {len(accepted)} patches (ballot-filtered)."
        )
        return {
            "path": "A",
            "candidate_path": "A",
            "candidate_source": "consensus_layer",
            "claude_override": False,
            "override_from": None,
            "override_to": None,
            "override_rule_check": {},
            "final_path": "A",
            "decision_reason": f"실험 실패 ({status}) -> 구현 수정 필요",
            "justification":   f"실험 실패 ({status})",
            "consensus_level": consensus.get("agreement_level", "weak"),
            "improvement_hints": hints,
            "evidence_strength": "low",
            "accepted_patch_indexes": accepted,
            "rejected_patch_indexes": [i for i in range(len(patches)) if i not in set(accepted)],
            "accepted_gpt_patches":   accepted,
            "accepted_patch_source": accepted_patch_source,
        }

    # ── Stage 7 ballot 정보 섹션 ─────────────────────────────
    ballot_section = ""
    if stage7_ballot_decisions:
        ballot_section = f"""
## Stage 7 Patch Ballot 결과 (accepted_patch_indexes 기준)
합의 수락: {ballot_accepted_indexes}
총 패치: {len(stage7_ballot_decisions)}개
상세: {json.dumps(stage7_ballot_decisions, ensure_ascii=False, indent=2)}

주의: accepted_patch_indexes는 Stage 7 ballot 결과에서 수락된 패치만 포함해야 합니다.
새로운 패치를 추가 승인하지 마세요. 기존 수락 패치에서 제외(subset)만 허용됩니다.
"""

    # ── Claude override reviewer prompt ──────────────────────
    prompt = f"""당신은 연구 파이프라인의 **override reviewer(Claude 역할)**입니다.
**중요: 당신은 자유롭게 path를 선택하는 것이 아닙니다.**
합의 레이어가 결정한 candidate_path="{candidate_path}"가 기본값입니다.
당신의 역할은 override가 정당한지 검토하는 것입니다.

## 합의 기본 경로 (candidate_path)
- candidate_path: {candidate_path}
- candidate_source: consensus_layer
- agreement_level: {agreement_lvl}

## Override 규칙 (엄격 적용)
1. candidate_path가 "done"이면 override 금지.
2. 허용되는 override: A->B, B->C (escalation만)
3. 금지되는 override: A->C (skip), B->A (downgrade), C->A/B (downgrade)
4. A->B override 조건 (모두 충족 필요):
   - GPT가 B 또는 C를 제안 (현재: {gpt_path})
   - Gemini가 strong disagree가 아님 (현재: {gem_agree})
   - agreement_level >= medium (현재: {agreement_lvl})
   - n_runs >= 2 (현재: {n_runs})
5. B->C override 조건 (모두 충족 필요):
   - agreement_level == strong (현재: {agreement_lvl})
   - evidence_strength == high (현재: {gpt_ev})
   - n_runs >= 3 (현재: {n_runs})
   - escalation_risk != blocked (현재: {consensus.get('escalation_risk', 'low')})

override 조건을 충족하지 못하면 candidate_path를 그대로 유지하세요.

## 실험 이력 ({n_runs}회)
{json.dumps(run_history, ensure_ascii=False, indent=2)}

## 현재 가설
{json.dumps(hypothesis.get('hypothesis', {}), ensure_ascii=False, indent=2)}

## GPT 심층 해석 (주요 해석자)
- suggested_path: {gpt_path}
- evidence_strength: {gpt_ev}
- confidence: {gpt_interpretation.get('confidence')}
- root_cause_analysis: {gpt_interpretation.get('root_cause_analysis', '')}
- hypothesis_validity: {gpt_interpretation.get('hypothesis_validity_assessment', '')}

## Gemini 2차 진단 (독립 의견)
- suggested_path: {gemini_diagnosis.get('suggested_path')}
- agreement_with_gpt: {gem_agree}
- disagreement_reason: {gemini_diagnosis.get('disagreement_reason', '')}

## 합의 상세
- escalation_risk: {consensus.get('escalation_risk')}
- major_disagreements: {consensus.get('major_disagreements', [])}
- notes_for_claude: {consensus.get('notes_for_claude', '')}

## GPT 패치 제안 ({len(patches)}개 — Path A 시에만 적용)
{json.dumps(patches, ensure_ascii=False, indent=2)}

## Hypothesis Implementation Audit
{json.dumps(summary.get('hypothesis_implementation', {}), ensure_ascii=False, indent=2)}

## Deterministic Diagnostic Prior (summary layer — rule-based, for reference)
- bottleneck_candidates: {json.dumps(summary.get('bottleneck_candidates', []), ensure_ascii=False)}
- recommended_next_actions: {json.dumps(summary.get('recommended_next_actions', []), ensure_ascii=False)}
- confidence_model: {json.dumps(summary.get('confidence_model', {}), ensure_ascii=False)}
- ablation_findings: {json.dumps(summary.get('ablation_findings', []), ensure_ascii=False)}
{ballot_section}

## 가설 구현 감사 규칙
- mechanism_audit.implemented=false -> Path A 우선
- mechanism 미구현 상태에서는 Path C 절대 금지

## accepted_patch_indexes 규칙
- Stage 7 ballot에서 수락된 패치: {ballot_accepted_indexes}
- 이 목록에서 subset만 선택 가능 (새 패치 추가 승인 금지)
- 제외 시 이유를 반드시 기재

아래 JSON으로만 출력:
{{
  "path": "A|B|C|done",
  "claude_override": false,
  "override_reason": "override 사유 (override=false이면 빈 문자열)",
  "decision_reason": "...",
  "consensus_level": "weak|medium|strong",
  "improvement_hints": "...",
  "evidence_strength": "low|medium|high",
  "accepted_patch_indexes": [0, 1],
  "rejected_patch_indexes": [2],
  "justification": "..."
}}"""

    try:
        result = parse_json(query_claude(prompt))

        claude_requested_path = result.get("path", candidate_path)
        claude_override = result.get("claude_override", claude_requested_path != candidate_path)

        # ── Override 규칙 검증 (Task 7) ─────────────────────────
        override_rule_check: dict = {}
        override_allowed = False

        if claude_override and claude_requested_path != candidate_path:
            override_from = candidate_path
            override_to = claude_requested_path

            # done -> override 금지
            if candidate_path == "done":
                override_allowed = False
                override_rule_check["blocked"] = "done cannot be overridden"

            # escalation만 허용: A->B, B->C
            elif override_from == "A" and override_to == "B":
                gpt_supports = gpt_path in ("B", "C")
                gemini_not_disagree = gem_agree != "disagree"
                agreement_ok = agreement_lvl in ("medium", "strong")
                runs_ok = n_runs >= 2
                override_allowed = gpt_supports and gemini_not_disagree and agreement_ok and runs_ok
                override_rule_check = {
                    "gpt_support": gpt_supports,
                    "gemini_support": gemini_not_disagree,
                    "agreement_level": agreement_lvl,
                    "run_count_ok": runs_ok,
                }

            elif override_from == "B" and override_to == "C":
                agreement_strong = agreement_lvl == "strong"
                evidence_high = gpt_ev == "high"
                runs_ok = n_runs >= 3
                esc_ok = consensus.get("escalation_risk") != "blocked"
                override_allowed = agreement_strong and evidence_high and runs_ok and esc_ok
                override_rule_check = {
                    "agreement_strong": agreement_strong,
                    "evidence_high": evidence_high,
                    "run_count_ok": runs_ok,
                    "escalation_not_blocked": esc_ok,
                }

            else:
                # 금지된 override (A->C, B->A, C->A/B, etc.)
                override_allowed = False
                override_rule_check["blocked"] = f"forbidden override direction: {override_from}->{override_to}"

            if not override_allowed:
                # override 불허 -> candidate_path 유지
                print(f"    [override 차단] Claude 요청 {override_from}->{override_to} 차단: {override_rule_check}")
                claude_requested_path = candidate_path
                claude_override = False
        else:
            claude_override = False

        final_path = claude_requested_path

        # ── accepted_patch_indexes: Stage 7 ballot subset 제한 (Task 8) ──
        claude_accepted = result.get("accepted_patch_indexes", ballot_accepted_indexes)
        ballot_set = set(ballot_accepted_indexes)
        # Claude는 ballot에서 수락된 패치의 subset만 선택 가능 (새 패치 추가 금지)
        filtered_accepted = [idx for idx in claude_accepted if idx in ballot_set]
        new_additions = [idx for idx in claude_accepted if idx not in ballot_set]
        if new_additions:
            print(f"    [패치 제한] Claude가 ballot 외 패치 추가 시도 차단: {new_additions}")
            accepted_patch_source_final = "claude_subset_override"
        else:
            accepted_patch_source_final = accepted_patch_source

        # 하위 호환 필드
        result["accepted_gpt_patches"]   = filtered_accepted
        result["rejected_gpt_patches"]   = result.get("rejected_patch_indexes", [])
        result.setdefault("justification", result.get("decision_reason", ""))

        # ── 구조화된 decision payload (Task 10) ──────────────────
        result.update({
            "path": final_path,
            "candidate_path": candidate_path,
            "candidate_source": "consensus_layer",
            "claude_override": claude_override,
            "override_from": candidate_path if claude_override else None,
            "override_to": final_path if claude_override else None,
            "override_rule_check": override_rule_check,
            "final_path": final_path,
            "accepted_patch_indexes": filtered_accepted,
            "accepted_patch_source": accepted_patch_source_final,
        })

        print(f"  [결정 / Claude] path={final_path}, "
              f"candidate={candidate_path}, override={claude_override}, "
              f"consensus={result.get('consensus_level')}")
        return result

    except Exception as e:
        print(f"    [경고] Claude 최종 결정 실패 ({e}) -> 합의 경로 기본 사용")
        gap          = primary["target"] - primary["value"]
        fallback_path = candidate_path
        accepted      = ballot_accepted_indexes if ballot_accepted_indexes else list(range(len(patches)))
        return {
            "path": fallback_path,
            "candidate_path": candidate_path,
            "candidate_source": "consensus_layer",
            "claude_override": False,
            "override_from": None,
            "override_to": None,
            "override_rule_check": {},
            "final_path": fallback_path,
            "decision_reason": f"{primary['name']}={primary['value']:.4f}, gap={gap:.4f} (fallback from consensus)",
            "justification":   f"{primary['name']}={primary['value']:.4f}, gap={gap:.4f}",
            "consensus_level": consensus.get("agreement_level", "weak"),
            "improvement_hints": (
                f"GPT: {gpt_interpretation.get('improvement_suggestions', [])}. "
                f"Patches: {len(patches)}개."
            ),
            "evidence_strength": gpt_interpretation.get("evidence_strength", "low"),
            "accepted_patch_indexes": accepted,
            "rejected_patch_indexes": [],
            "accepted_gpt_patches":   accepted,
            "rejected_gpt_patches":   [],
            "accepted_patch_source": accepted_patch_source,
        }


# ──────────────────────────────────────────────────────────
# Claude 결정 후 결정론적 사후 검증
# ──────────────────────────────────────────────────────────

def _postcheck_final_decision(
    decision: dict,
    consensus: dict,
    gpt_interpretation: dict,
    gemini_diagnosis: dict,
    run_history: list[dict],
    summary: dict,
) -> dict:
    """Claude 결정 후 코드 레벨에서 Path B/C 허용 조건을 재검증한다 (LLM 호출 없음).

    Task 9: override 최종 허가자 역할.
    - candidate_path, claude_override, override_from, override_to를 입력으로 검증
    - Claude override가 postcheck를 통과하지 못하면 candidate_path로 되돌림
    - mechanism 미구현 상태에서 Path C 절대 금지 규칙 유지

    반환 필드:
      path, decision_reason, consensus_level,
      postcheck_applied, postcheck_override, postcheck_note, postcheck_ok,
      override_rule_check, original_claude_path
    """
    path           = decision.get("path", "A")
    primary        = summary["primary_metric"]
    status         = summary["status"]
    n_runs         = len(run_history)
    gpt_path       = gpt_interpretation.get("suggested_path", "A")
    gpt_ev         = gpt_interpretation.get("evidence_strength", "low")
    gem_agree      = gemini_diagnosis.get("agreement_with_gpt", "agree")
    agreement_lvl  = consensus.get("agreement_level", "weak")
    consensus_path = consensus.get("consensus_path_candidate", "A")

    # consensus-locked 필드 추출
    candidate_path    = decision.get("candidate_path", consensus_path)
    claude_override   = decision.get("claude_override", False)
    override_from     = decision.get("override_from")
    override_to       = decision.get("override_to")

    # ── 허용 조건 함수 ──────────────────────────────────────

    def _b_allowed() -> tuple[bool, str]:
        """Path B 허용 조건. (허용여부, 실패 사유) 반환."""
        if gpt_path not in ("B", "C"):
            return False, f"GPT suggested {gpt_path}, not B/C"
        if gem_agree == "disagree":
            return False, "Gemini strongly disagrees"
        if agreement_lvl == "weak":
            return False, f"consensus too weak ({agreement_lvl})"
        if n_runs < 2:
            return False, f"insufficient runs (n_runs={n_runs}, need >=2)"
        if status in ("smoke_failed", "failed", "metrics_parse_error", "timeout") and n_runs < 2:
            return False, f"pure execution failure without prior success (status={status})"
        return True, ""

    def _c_allowed() -> tuple[bool, str]:
        """Path C 허용 조건. (허용여부, 실패 사유) 반환."""
        if n_runs < 3:
            return False, f"insufficient runs (n_runs={n_runs}, need >=3)"
        if agreement_lvl != "strong":
            return False, f"consensus not strong ({agreement_lvl})"
        if gpt_ev != "high":
            return False, f"GPT evidence_strength not high ({gpt_ev})"
        if consensus.get("escalation_risk") == "blocked":
            return False, "escalation_risk=blocked in consensus"
        # 구현/훈련 불안정 케이스는 C 불가
        stab = summary.get("training_stability", {})
        if stab.get("nan_detected") or not stab.get("loss_converged", True):
            return False, "training instability detected -- implementation issue, not hypothesis failure"
        # mechanism 미구현 상태에서는 Path C 절대 금지
        hyp_impl = summary.get("hypothesis_implementation", {})
        mech_audit = hyp_impl.get("mechanism_audit", {})
        if mech_audit and not mech_audit.get("implemented", True):
            return False, "mechanism not implemented in code -- must fix implementation before hypothesis rejection"
        return True, ""

    # ── 사후 검증 로직 ───────────────────────────────────────

    postcheck_override = False
    postcheck_note     = None
    postcheck_ok       = True
    original_path      = path

    # ── 1. done 조건 재확인 ──────────────────────────────────
    if path == "done":
        if not primary["met"]:
            corrected = candidate_path if candidate_path in ("A", "B") else "A"
            postcheck_override, postcheck_note, postcheck_ok = True, (
                f"Claude returned done but primary metric not met "
                f"({primary['name']}={primary['value']:.4f} < {primary['target']:.2f}) "
                f"-- corrected to {corrected}"
            ), False
            path = corrected

    # ── 2. Override 최종 허가 (Task 9) ───────────────────────
    elif claude_override and override_from and override_to:
        # override가 요청된 경우 postcheck가 최종 허가자 역할
        if override_to == "C":
            ok, reason = _c_allowed()
            if not ok:
                # Path C override 실패 -> candidate_path로 되돌림
                postcheck_override, postcheck_note, postcheck_ok = True, (
                    f"Override {override_from}->{override_to} denied by postcheck: {reason} "
                    f"-- reverted to candidate_path={candidate_path}"
                ), False
                path = candidate_path
        elif override_to == "B":
            ok, reason = _b_allowed()
            if not ok:
                postcheck_override, postcheck_note, postcheck_ok = True, (
                    f"Override {override_from}->{override_to} denied by postcheck: {reason} "
                    f"-- reverted to candidate_path={candidate_path}"
                ), False
                path = candidate_path

    # ── 3. Path C/B guardrail (override 아닌 경우에도 적용) ──
    elif path == "C":
        ok, reason = _c_allowed()
        if not ok:
            b_ok, _ = _b_allowed()
            corrected = "B" if b_ok else "A"
            postcheck_override, postcheck_note, postcheck_ok = True, (
                f"Path C not allowed: {reason} -- downgraded to {corrected}"
            ), False
            path = corrected

    elif path == "B":
        ok, reason = _b_allowed()
        if not ok:
            corrected = candidate_path if candidate_path == "A" else "A"
            postcheck_override, postcheck_note, postcheck_ok = True, (
                f"Path B not allowed: {reason} -- downgraded to {corrected}"
            ), False
            path = corrected

    # ── 결과 조립 (구조화된 decision payload — Task 10) ─────

    result = dict(decision)   # 원본 필드 모두 보존
    result["path"]              = path
    result["final_path"]        = path
    result["postcheck_applied"] = True
    result["postcheck_override"] = postcheck_override
    result["postcheck_ok"]      = postcheck_ok
    result["postcheck_note"]    = postcheck_note

    # override_rule_check 보강
    override_rule_check = result.get("override_rule_check", {})
    override_rule_check["postcheck_ok"] = postcheck_ok
    result["override_rule_check"] = override_rule_check

    if postcheck_override:
        result["original_claude_path"] = original_path
        # override가 postcheck에서 되돌려진 경우 claude_override를 무효화
        if not postcheck_ok and claude_override:
            result["claude_override"] = False
            result["override_from"] = None
            result["override_to"] = None
        note_prefix = f"[postcheck override: {original_path}->{path}] "
        result["decision_reason"] = note_prefix + result.get("decision_reason", "")
        result["justification"]   = note_prefix + result.get("justification", "")
        print(f"    [사후검증 !]  Claude={original_path} -> 수정={path}: {postcheck_note}")
    else:
        result["original_claude_path"] = original_path
        print(f"    [사후검증 OK] Claude={original_path} 유지 (postcheck_ok={postcheck_ok})")

    return result


# ──────────────────────────────────────────────────────────
# Stage 7 ballot 결과 로드 (Task 8, 11)
# ──────────────────────────────────────────────────────────

def _load_stage7_ballot_decisions(pkg_dir: Path) -> list[dict] | None:
    """패키지의 proposals/에서 가장 최근 merge_decision_*.json을 로드한다.

    merge_decision이 없으면 None 반환 (하위 호환 — ballot 없이 동작).
    """
    proposals_dir = pkg_dir / "proposals"
    if not proposals_dir.exists():
        return None

    # merge_decision_*.json 파일을 타임스탬프 기준으로 정렬, 최신 파일 사용
    candidates = sorted(
        proposals_dir.glob("merge_decision_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None

    try:
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        decisions = data.get("decisions", [])
        if decisions:
            print(f"    [Stage 7 ballot] {candidates[0].name} 로드: "
                  f"{len(decisions)}개 패치 결정")
        return decisions if decisions else None
    except Exception as e:
        print(f"    [경고] Stage 7 ballot 로드 실패: {e}")
        return None


# ──────────────────────────────────────────────────────────
# revision_request.json 생성
# ──────────────────────────────────────────────────────────

def _write_revision_request(
    reports_dir: Path,
    decision: dict,
    summary: dict,
    run_history: list[dict],
    hypothesis: dict,
    version: int,
) -> Path:
    slug = summary["run_id"].rsplit("_v", 1)[0]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = decision["path"]

    req = {
        "schema_version":      "1.0",
        "revision_id":         f"rev_{path}_{slug}_{ts}",
        "path":                path,
        "triggered_by_run_id": summary["run_id"],
        "supporting_run_ids":  [r["run_id"] for r in run_history],
        "hypothesis_id":       summary["hypothesis_id"],
        "experiment_version":  version,
        "justification":       decision["justification"],
        "evidence_strength":   decision["evidence_strength"],
        "falsification_evidence": decision.get("falsification_evidence"),
        "artifacts_may_change": (
            ["model.py", "module.py", "data.py", "configs/default.yaml"]
            if path == "A" else
            ["hypothesis.json", "experiment_plan.json"]
        ),
        "artifacts_must_stay": (
            ["hypothesis.json", "experiment_plan.json", "approval.json"]
            if path == "A" else
            ["approval.json", "previous_results.jsonl"]
        ),
        "comparability_note":      decision.get("improvement_hints", ""),
        "next_experiment_version": version + 1,
        # consensus-locked decision 필드
        "candidate_path":          decision.get("candidate_path"),
        "candidate_source":        decision.get("candidate_source", "consensus_layer"),
        "claude_override":         decision.get("claude_override", False),
        "override_from":           decision.get("override_from"),
        "override_to":             decision.get("override_to"),
        "override_rule_check":     decision.get("override_rule_check", {}),
        "accepted_patch_source":   decision.get("accepted_patch_source"),
        # postcheck 감사 필드 — postcheck_override=True이면 원본 Claude 결정과 다름
        "postcheck_override":      decision.get("postcheck_override", False),
        "postcheck_ok":            decision.get("postcheck_ok", True),
        "original_claude_path":    decision.get("original_claude_path"),
        "postcheck_note":          decision.get("postcheck_note"),
        "created_at":              datetime.now().isoformat(),
    }

    if path == "B":
        req["path_B_details"] = {
            "original_claim":   hypothesis.get("hypothesis", {}).get("statement_kr", ""),
            "refined_claim":    "— 가설 정제 필요 —",
            "narrowing_reason": decision["justification"],
            "preserved_core":   "핵심 메커니즘 유지",
        }
    elif path == "C":
        req["path_C_details"] = {
            "supersedes_hypothesis_id": summary["hypothesis_id"],
            "negative_evidence_runs":   [r["run_id"] for r in run_history],
            "proposed_new_direction":   decision.get("improvement_hints", ""),
        }

    req_path = reports_dir / f"revision_request_v{version}.json"
    req_path.write_text(json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [저장] {req_path}")
    return req_path


# ──────────────────────────────────────────────────────────
# previous_results.jsonl append
# ──────────────────────────────────────────────────────────

def _sanitize_summary(summary: dict) -> dict:
    """result_summary에서 민감 정보를 제거한다."""
    import re
    _SECRET_RE = re.compile(
        r"(ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|"
        r"github_pat_[A-Za-z0-9_]{82}|"
        r"sk-[A-Za-z0-9]{48,}|"
        r"AIza[A-Za-z0-9\-_]{35})"
    )
    text = json.dumps(summary, ensure_ascii=False)
    sanitized = _SECRET_RE.sub("***REDACTED***", text)
    return json.loads(sanitized)


def _append_results_log(summary: dict, pkg_dir: Path) -> None:
    # experiments/{slug}/results/previous_results.jsonl
    slug = slug_from_pkg(pkg_dir)
    res_dir = results_dir(slug)
    res_dir.mkdir(parents=True, exist_ok=True)
    log_path = res_dir / "previous_results.jsonl"
    clean = _sanitize_summary(summary)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(f"    [append] {log_path} (sanitized)")


# ──────────────────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# GitHub Issue 알림 (실험 완료/실패 시 자동 생성)
# ──────────────────────────────────────────────────────────

_INFRA_FAILURE_STATUSES = {"smoke_failed", "failed", "timeout", "metrics_parse_error"}


def _is_infra_failure(summary: dict) -> tuple[bool, str]:
    """실험 실패가 인프라/코드 문제인지 판단한다.

    Returns: (is_infra, reason)
    - True: 코드 버그, 시스템 에러 → 즉시 중단
    - False: 성능 미달 → Path A 재시도 가능
    """
    status = summary.get("status", "")
    if status in _INFRA_FAILURE_STATUSES:
        stderr = summary.get("stderr_tail", [])
        stderr_text = " ".join(stderr[-5:]) if stderr else ""
        # smoke_failed의 구체적 원인 분류
        if "ModuleNotFoundError" in stderr_text or "ImportError" in stderr_text:
            return True, f"import error: {stderr_text[:200]}"
        if "SyntaxError" in stderr_text:
            return True, f"syntax error: {stderr_text[:200]}"
        if "NameError" in stderr_text or "AttributeError" in stderr_text:
            return True, f"code error: {stderr_text[:200]}"
        if "CUDA" in stderr_text or "out of memory" in stderr_text.lower():
            return True, f"GPU/memory error: {stderr_text[:200]}"
        if "git" in stderr_text.lower() and "push" in stderr_text.lower():
            return True, f"git push error: {stderr_text[:200]}"
        if "timeout" in status:
            return True, f"execution timeout"
        # smoke_failed이지만 구체적 원인 불명 → 인프라 문제로 간주
        if status == "smoke_failed":
            return True, f"smoke test failed: {stderr_text[:200]}"
        return True, f"execution failed: {status}"
    return False, ""


def _create_github_issue(title: str, body: str, labels: list[str] | None = None) -> str:
    """GitHub Issue를 생성하여 사용자에게 알린다.

    Returns: issue URL or error message
    """
    import requests
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    owner = os.environ.get("GITHUB_OWNER", "").strip()
    repo = os.environ.get("GITHUB_REPO", "").strip()

    if not (token and owner and repo):
        msg = f"[알림] GitHub Issue 생성 불가 (env 미설정) — {title}"
        print(msg)
        return msg

    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data = {"title": title, "body": body, "labels": labels or []}

    try:
        resp = requests.post(url, json=data, headers=headers, timeout=30)
        if resp.status_code == 201:
            issue_url = resp.json().get("html_url", "")
            print(f"  [알림] GitHub Issue 생성: {issue_url}")
            return issue_url
        else:
            msg = f"[알림] Issue 생성 실패 ({resp.status_code}): {resp.text[:200]}"
            print(msg)
            return msg
    except Exception as e:
        msg = f"[알림] Issue 생성 오류: {e}"
        print(msg)
        return msg


def _notify_experiment_result(
    summary: dict, decision: dict | None, slug: str, version: int,
    is_infra_fail: bool = False, infra_reason: str = "",
) -> None:
    """실험 결과를 GitHub Issue로 알린다."""
    primary = summary.get("primary_metric", {})
    status = summary.get("status", "")
    met = primary.get("met", False)

    if is_infra_fail:
        # 인프라/코드 실패 → 즉시 중단 알림
        title = f"🔴 실험 실패 (v{version}): {slug} — 인프라/코드 오류"
        body = f"""## 실험 실패 — 즉시 확인 필요

| 항목 | 값 |
|------|-----|
| **실험** | `{slug}` v{version} |
| **상태** | `{status}` |
| **원인** | {infra_reason} |
| **metric** | {primary.get('name', '?')} = {primary.get('value', 0)} |

### 조치 필요
- 코드/환경 문제를 수정 후 재실행하세요
- Path A 자동 재시도는 **중단**되었습니다

### stderr (마지막 5줄)
```
{chr(10).join(summary.get('stderr_tail', [])[-5:])}
```
"""
        _create_github_issue(title, body, labels=["bug", "experiment-failed"])

    elif met:
        # 목표 달성 → 완료 알림
        path = decision.get("path", "done") if decision else "done"
        title = f"🟢 실험 완료 (v{version}): {slug} — 목표 달성!"
        body = f"""## 실험 완료 — 목표 달성

| 항목 | 값 |
|------|-----|
| **실험** | `{slug}` v{version} |
| **{primary.get('name', '?')}** | **{primary.get('value', 0):.4f}** |
| **목표** | {primary.get('target', 0)} |
| **판정** | ✅ 달성 |
| **confidence** | {summary.get('confidence', 0)} |

### 다음 단계
결과를 확인하고 논문 작성 또는 추가 실험을 진행하세요.
- `experiments/{slug}/results/v{version}/result_summary.json`
- `experiments/{slug}/reports/report.pdf`
"""
        _create_github_issue(title, body, labels=["experiment-done"])


def run_research_loop(
    pkg_dir: str,
    topic_file: str,
    hypothesis_file: str,
    code_analysis_file: str,
    max_rounds: int = 3,
    config_file: str = "configs/default.yaml",
    runner_type: str = "github",
    runner_config: dict | None = None,
) -> dict:
    """실험 실행 + Path A/B/C revision 루프. 최종 result_summary 반환."""
    topic       = json.loads(Path(topic_file).read_text(encoding="utf-8"))
    hypothesis  = json.loads(Path(hypothesis_file).read_text(encoding="utf-8"))
    reports_dir = Path(topic_file).parent

    pkg     = Path(pkg_dir)
    # pkg.name = "v1", "v2", ... (new structure only — 과거 구조는 에러 처리)
    version = version_from_pkg(pkg)

    # runner 초기화 + 준비 상태 확인 (미구현이면 즉시 종료)
    runner: BaseRunner = create_runner(runner_type, runner_config)
    _check_runner_ready(runner)
    print(f"  [Runner] {runner.__class__.__name__} 사용")

    run_history:     list[dict] = []
    prev_run_id:     str | None = None
    prev_metrics:    dict       = {}
    final_summary:   dict       = {}
    final_decision:  dict | None = None

    for round_idx in range(1, max_rounds + 1):
        print(f"\n{'─'*60}")
        print(f"  [연구 루프] Round {round_idx}/{max_rounds} — {pkg.name}")
        print(f"{'─'*60}")

        # 구조 검증
        missing = _validate_package(pkg)
        if missing:
            print(f"  [오류] 필수 파일 누락: {missing}")
            break

        # smoke test
        smoke_result = runner.run_smoke(pkg)
        smoke_ok = smoke_result["status"] == "success"

        if not smoke_ok:
            run_result = {
                "status": "smoke_failed", "metrics": {},
                "stdout_lines": smoke_result["stdout_lines"],
                "stderr_tail":  smoke_result["stderr_tail"],
                "returncode":   smoke_result["returncode"],
                "metadata":     smoke_result["metadata"],
            }
        else:
            run_result = runner.run_train(pkg, config_file)

        # spec 로드
        spec_path = pkg / "experiment_spec.json"
        if spec_path.exists():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        else:
            inp  = topic.get("input", {})
            slug = re.sub(r"\W+", "_", inp.get("topic", "research").lower())[:30]
            # target_metric → 첫 번째 지표를 primary로 사용 (없으면 빈 문자열)
            raw_metric = inp.get("target_metric", "")
            primary_metric = re.split(r"[,\s]+", raw_metric.strip())[0].lower() if raw_metric else ""
            spec = {
                "schema_version": "1.0", "spec_id": f"{slug}_v{version}",
                "hypothesis_id": str(uuid.uuid4())[:8],
                "experiment_version": version, "topic_slug": slug,
                "evaluation_config": {
                    "primary_metric": primary_metric, "target_value": 0.0,
                    "secondary_metrics": [], "test_set": "val_split",
                },
            }

        # result_summary 생성 + 기록 (run_history passed for A-5 trend + A-6 ablation)
        summary = _build_result_summary(pkg, run_result, spec, prev_run_id, prev_metrics, run_history)
        _append_results_log(summary, pkg)
        final_summary = summary

        primary = summary["primary_metric"]
        print(f"\n  [결과] {primary['name']} = {primary['value']:.4f} "
              f"(target={primary['target']:.2f}, met={primary['met']})")

        # ── 인프라/코드 실패 감지 → 즉시 중단 + 알림 ──
        is_infra, infra_reason = _is_infra_failure(summary)
        if is_infra:
            print(f"\n  🔴 [인프라 실패] {infra_reason}")
            print(f"     Path A 재시도 없이 즉시 중단합니다.")
            _notify_experiment_result(
                summary, None, slug_from_pkg(pkg), version,
                is_infra_fail=True, infra_reason=infra_reason,
            )
            final_summary = summary
            break

        stability = summary.get("training_stability", {})
        hyp_impl = summary.get("hypothesis_implementation", {})
        mech_audit = hyp_impl.get("mechanism_audit", {})
        metric_audit = hyp_impl.get("metric_audit", {})
        constraints_audit = hyp_impl.get("constraints_audit", {})

        # A-7: Richer run_history entries
        run_entry: dict = {
            # 기본 실행 정보
            "run_id":  summary["run_id"],
            "version": version,
            "status":  summary["status"],
            "metrics": summary["primary_metric"],
            # A-7: full training stability (including plateau, overfit, etc.)
            "training_stability": stability,
            # A-7: implementation audit flags
            "hypothesis_implementation_ok": bool(
                (not mech_audit or mech_audit.get("implemented", True))
                and (not metric_audit or metric_audit.get("correct", True))
                and (not constraints_audit or constraints_audit.get("satisfied", True))
            ),
            "mechanism_ok":    not mech_audit or mech_audit.get("implemented", True),
            "metric_ok":       not metric_audit or metric_audit.get("correct", True),
            "constraints_ok":  not constraints_audit or constraints_audit.get("satisfied", True),
            # A-7: confidence from model
            "confidence":      summary.get("confidence", 0.0),
            # A-7: bottleneck candidates and recommended actions
            "bottleneck_candidates":     summary.get("bottleneck_candidates", []),
            "recommended_next_actions":  summary.get("recommended_next_actions", []),
            # A-7: patch family tags (populated after patch phase)
            "patch_family_tags":         [],
            # A-7: ablation findings
            "ablation_findings":         summary.get("ablation_findings", []),
            # 결정 컨텍스트 — 분석 완료 후 업데이트됨
            "decision_path":        None,
            "consensus_level":      None,
            "gpt_suggested_path":   None,
            "gemini_suggested_path": None,
            "accepted_patch_indexes": [],
            "blocked_reason":       None,
            "validation_failed":    False,
        }
        run_history.append(run_entry)
        prev_run_id  = summary["run_id"]
        prev_metrics = {
            primary["name"]: primary["value"],
            **{m["name"]: m["value"] for m in summary["secondary_metrics"]},
        }

        # ── Multi-model 분석 ───────────────────────────────
        # 파이프라인: GPT(해석) -> Gemini(진단) -> 합의 -> GPT(패치) -> Claude(override review)
        print(f"\n  [분석] GPT(해석) -> Gemini(진단) -> 합의 -> GPT(패치) -> Claude(override review)")

        # 1. GPT: 주요 해석자
        gpt_interp    = _gpt_interpret_results(summary, hypothesis, spec, run_history)
        gpt_interp_path = _save_proposal(pkg, "gpt_interpretation", gpt_interp)

        # 2. Gemini: 독립 2차 진단
        gemini_diag   = _gemini_short_diagnosis(summary, hypothesis, gpt_interp)
        gemini_diag_path = _save_proposal(pkg, "gemini_diagnosis", gemini_diag)

        # 3. 합의 레이어
        consensus     = _build_consensus(gpt_interp, gemini_diag, run_history, primary)

        # 4. GPT: 패치 제안 — consensus_path_candidate == "A" 일 때만 실행
        # Path B/C는 가설 정제/교체 영역이므로 코드 패치 생성 불필요
        candidate_path = consensus.get("consensus_path_candidate", "A")
        if not primary["met"] and candidate_path == "A":
            gpt_patches    = _gpt_propose_improvements(pkg, summary, gpt_interp)
            gpt_patch_path = _save_proposal(pkg, "gpt_patch_result", gpt_patches)
        else:
            gpt_patches    = {"patches": [], "confidence": "n/a",
                              "skipped_reason": f"consensus_path={candidate_path} (patches only for A)"}
            gpt_patch_path = None

        # 4-B. Stage 7 ballot 결과 로드 (Task 8, 11)
        stage7_ballot = _load_stage7_ballot_decisions(pkg)

        # 5. Claude: override reviewer (합의 경로 기본, override는 escalation만)
        decision = _claude_final_decision(
            pkg, summary, hypothesis, run_history,
            gpt_interp, gemini_diag, consensus, gpt_patches, max_rounds,
            stage7_ballot_decisions=stage7_ballot,
        )

        # 6. 결정론적 사후 검증 — Path B/C guardrail 재확인 (LLM 없음)
        decision = _postcheck_final_decision(
            decision, consensus, gpt_interp, gemini_diag, run_history, summary,
        )

        # proposal archive
        _archive_proposal(gpt_interp_path, True)
        _archive_proposal(gemini_diag_path, True)
        if gpt_patch_path:
            accepted_indices = set(decision.get("accepted_patch_indexes", []))
            _archive_proposal(gpt_patch_path, bool(accepted_indices))

        # Task 11: 구조화된 decision payload 저장
        _save_proposal(pkg, "consensus_decision", {
            "candidate_path":        decision.get("candidate_path"),
            "candidate_source":      decision.get("candidate_source", "consensus_layer"),
            "claude_override":       decision.get("claude_override", False),
            "override_from":         decision.get("override_from"),
            "override_to":           decision.get("override_to"),
            "override_rule_check":   decision.get("override_rule_check", {}),
            "final_path":            decision.get("final_path", decision.get("path")),
            "accepted_patch_indexes": decision.get("accepted_patch_indexes", []),
            "accepted_patch_source": decision.get("accepted_patch_source", "unknown"),
            "decision_reason":       decision.get("decision_reason", ""),
            "justification":         decision.get("justification", ""),
            "consensus_level":       decision.get("consensus_level", ""),
            "evidence_strength":     decision.get("evidence_strength", ""),
            "postcheck_ok":          decision.get("postcheck_ok", True),
            "postcheck_override":    decision.get("postcheck_override", False),
            "postcheck_note":        decision.get("postcheck_note"),
            "created_at":            datetime.now().isoformat(),
        })

        # A-7: Classify accepted patches into family tags
        accepted_indices = decision.get("accepted_patch_indexes", [])
        all_patches = gpt_patches.get("patches", [])
        accepted_patches = [all_patches[i] for i in accepted_indices if i < len(all_patches)]
        patch_family_tags = _classify_patch_families(accepted_patches)

        # run_history 마지막 항목에 결정 컨텍스트 보강 (사후검증 결과 + consensus-locked 필드 포함)
        run_history[-1].update({
            "decision_path":         decision.get("path"),
            "candidate_path":        decision.get("candidate_path"),
            "claude_override":       decision.get("claude_override", False),
            "consensus_level":       decision.get("consensus_level", consensus.get("agreement_level")),
            "gpt_suggested_path":    gpt_interp.get("suggested_path"),
            "gemini_suggested_path": gemini_diag.get("suggested_path"),
            "accepted_patch_indexes": accepted_indices,
            "accepted_patch_source": decision.get("accepted_patch_source"),
            "postcheck_override":    decision.get("postcheck_override", False),
            "postcheck_ok":          decision.get("postcheck_ok", True),
            "original_claude_path":  decision.get("original_claude_path"),
            "postcheck_note":        decision.get("postcheck_note"),
            # A-7: patch family tags
            "patch_family_tags":     patch_family_tags,
        })

        final_decision = decision
        print(f"  [결정] Path {decision['path']} — {decision['justification'][:80]}")

        if decision["path"] == "done":
            print(f"\n  🎉 목표 달성! {primary['name']}={primary['value']:.4f}")
            _notify_experiment_result(
                summary, decision, slug_from_pkg(pkg), version,
            )
            break

        if decision["path"] == "A" and round_idx < max_rounds:
            # Path A: 이전 패키지 기반 최소 revision으로 다음 버전 생성
            version += 1
            print(f"\n  [Path A] v{version} 패키지 revision (이전 패키지 기반 최소 diff)...")

            # 원본 GPT 패치 전체 + 원본 인덱스를 그대로 전달
            # (재번호 매기기 없이 _prepare_accepted_patch_context가 원본 인덱스를 보존)
            all_patches   = gpt_patches.get("patches", [])
            accepted_idxs = sorted(
                i for i in decision.get("accepted_patch_indexes", decision.get("accepted_gpt_patches", []))
                if i < len(all_patches)
            )

            gen = generate_experiment_package(
                topic_file            = topic_file,
                hypothesis_file       = hypothesis_file,
                code_analysis_file    = code_analysis_file,
                version               = version,
                revised_from          = str(pkg),
                revision_path         = "A",
                improvement_hints     = decision.get("improvement_hints", ""),
                accepted_gpt_patches  = all_patches,       # 전체 패치 목록 (원본 순서)
                accepted_gpt_indexes  = accepted_idxs,     # 원본 인덱스 그대로
                gemini_review_ctx     = gpt_interp,        # GPT 해석이 메인 컨텍스트
                result_summary_ctx    = summary,
            )

            # validation gate: finalized=False 이면 루프 중단
            if not gen.get("finalized", True):
                blocked = gen.get("blocked_reason", "unknown")
                print(f"\n  [루프 중단] Path A 패키지 finalization 차단: {blocked}")
                print(f"     (생성된 파일은 디버깅용으로 {gen['pkg_dir']}에 보존됨)")
                # 현재 run 항목에 차단 사유 기록
                run_history[-1]["blocked_reason"]    = blocked
                run_history[-1]["validation_failed"] = gen.get("validation_failed", True)
                _write_revision_request(
                    reports_dir, decision, summary, run_history, hypothesis, version,
                )
                break

            pkg = Path(gen["pkg_dir"])
            config_file = "configs/default.yaml"
            continue

        # Path A(마지막) / B / C: revision_request 생성 후 종료
        _write_revision_request(
            reports_dir, decision, summary, run_history, hypothesis, version
        )
        print(f"\n  [종료] Path {decision['path']} — 상위 파이프라인에서 처리 필요")
        break

    # ── 결과 보고서 PDF 생성 (통합 PDF: 가설 + 실험결과 + 참고문헌) ──
    if final_summary:
        from lab.result_report import generate_result_report
        # 구버전 result_report.pdf (호환성 유지)
        generate_result_report(
            pkg_dir=pkg,
            final_summary=final_summary,
            run_history=run_history,
            topic_file=topic_file,
            hypothesis_file=hypothesis_file,
            decision=final_decision,
        )
        # 통합 report.pdf (가설 + 결과 + 분석 + 참고문헌)
        try:
            from lab.user_approval import generate_pdf
            from lab.config import slug_from_pkg, reports_dir as _rdir
            _topic = json.loads(Path(topic_file).read_text(encoding="utf-8"))
            _hyp   = json.loads(Path(hypothesis_file).read_text(encoding="utf-8"))
            _slug  = slug_from_pkg(pkg)
            _rdir_path = _rdir(_slug)
            _papers_path = _rdir_path / "papers.json"
            _val_path    = _rdir_path / "validation.json"
            _papers = json.loads(_papers_path.read_text(encoding="utf-8")) if _papers_path.exists() else {}
            _val    = json.loads(_val_path.read_text(encoding="utf-8")) if _val_path.exists() else {}
            _prev_path = Path(f"experiments/{_slug}/results/previous_results.jsonl")
            _results = []
            if _prev_path.exists():
                for line in _prev_path.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        _results.append(json.loads(line))
            generate_pdf(_topic, _hyp, _papers, _val,
                         _rdir_path / "report.pdf",
                         results=_results, final_summary=final_summary)
        except Exception as e:
            print(f"    [PDF 경고] 통합 보고서 생성 실패: {e}")

    # ── 실험 결과 git push ────────────────────────────────
    if final_summary:
        _push_results(slug_from_pkg(pkg), version_from_pkg(pkg))

    return final_summary


def _push_results(slug: str, version: int) -> None:
    """실험 결과(results/, reports/, runs/)를 git commit & push한다.

    체크포인트 등 대용량 파일은 .gitignore에서 제외됨.
    push 실패 시 경고만 출력하고 파이프라인은 계속 진행.
    """
    exp_dir = f"experiments/{slug}"
    try:
        # git config 보장 (container 내 미설정 시 fallback)
        subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True,
        ).returncode != 0 and subprocess.run(
            ["git", "config", "user.email", "pipeline@research.local"],
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True,
        ).returncode != 0 and subprocess.run(
            ["git", "config", "user.name", "research-pipeline"],
            capture_output=True,
        )

        # stage: results + reports + runs (코드 변경분)
        subprocess.run(
            ["git", "add", f"{exp_dir}/results/", f"{exp_dir}/reports/",
             f"{exp_dir}/runs/"],
            capture_output=True, text=True,
        )
        # 변경사항 확인
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True,
        )
        if diff.returncode == 0:
            print("    [git] 변경사항 없음, push 건너뜀")
            return

        subprocess.run(
            ["git", "commit", "-m",
             f"experiment: {slug} v{version} results"],
            capture_output=True, text=True, check=True,
        )
        # remote 자동 감지 (origin 우선, 없으면 첫 번째 remote)
        remotes = subprocess.run(
            ["git", "remote"], capture_output=True, text=True,
        ).stdout.strip().split("\n")
        remote = "origin" if "origin" in remotes else remotes[0] if remotes else ""
        if not remote:
            print("    [git 경고] remote 없음, push 건너뜀")
            return

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()

        # push 전 pull --rebase (원격이 더 최신일 수 있음)
        pull = subprocess.run(
            ["git", "pull", "--rebase", remote, branch],
            capture_output=True, text=True,
        )
        if pull.returncode != 0:
            print(f"    [git 경고] pull --rebase 실패: {pull.stderr.strip()[:200]}")

        push = subprocess.run(
            ["git", "push", remote, branch],
            capture_output=True, text=True,
        )
        if push.returncode == 0:
            print(f"    [git] 결과 push 완료 → {remote}/{branch}")
        else:
            print(f"    [git 경고] push 실패: {push.stderr.strip()[:200]}")
    except Exception as e:
        print(f"    [git 경고] 결과 push 중 오류: {e}")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Loop — 실험 실행 + revision")
    parser.add_argument("--pkg-dir",         required=True,
                        help="experiments/{slug}/runs/v{N} 패키지 디렉토리")
    parser.add_argument("--topic-file",      required=True)
    parser.add_argument("--hypothesis-file", required=True)
    parser.add_argument("--code-file",       required=True)
    parser.add_argument("--max-rounds",      type=int, default=3)
    parser.add_argument("--config-file",     default="configs/default.yaml")
    # Runner 선택
    parser.add_argument("--runner-type",     default="github", choices=["local", "github"],
                        help="실험 실행 방식 (기본값: local)")
    # GitHub Actions runner 설정 (--runner-type github 시 사용)
    parser.add_argument("--github-token",    default=os.getenv("GITHUB_TOKEN", ""))
    parser.add_argument("--github-owner",    default=os.getenv("GITHUB_OWNER", ""))
    parser.add_argument("--github-repo",     default=os.getenv("GITHUB_REPO", ""))
    parser.add_argument("--github-ref",      default=os.getenv("GITHUB_REF", "main"))
    parser.add_argument("--github-workflow", default=os.getenv("GITHUB_WORKFLOW", "experiment.yml"))
    args = parser.parse_args()

    runner_config = {
        "github_token":    args.github_token,
        "github_owner":    args.github_owner,
        "github_repo":     args.github_repo,
        "github_ref":      args.github_ref,
        "github_workflow": args.github_workflow,
    }

    result = run_research_loop(
        pkg_dir            = args.pkg_dir,
        topic_file         = args.topic_file,
        hypothesis_file    = args.hypothesis_file,
        code_analysis_file = args.code_file,
        max_rounds         = args.max_rounds,
        config_file        = args.config_file,
        runner_type        = args.runner_type,
        runner_config      = runner_config,
    )

    print(f"\n{'='*60}")
    print(f"  연구 루프 완료")
    primary = result.get("primary_metric", {})
    print(f"  최종: {primary.get('name','?')} = {primary.get('value','?')} "
          f"(목표: {primary.get('target','?')}, 달성: {primary.get('met','?')})")
    print(f"{'='*60}")
