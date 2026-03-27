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
# result_summary.json 생성
# ──────────────────────────────────────────────────────────

def _build_result_summary(
    pkg_dir: Path,
    run_result: dict,
    spec: dict,
    previous_run_id: str | None,
    previous_metrics: dict,
) -> dict:
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

    stdout_text  = "\n".join(run_result.get("stdout_lines", []))
    nan_detected = "nan" in stdout_text.lower() or "inf" in stdout_text.lower()
    converged    = run_result["status"] == "success" and bool(run_result["metrics"])

    run_id = (
        f"{spec['topic_slug']}_v{spec['experiment_version']}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # runner metadata (runner abstraction에서 제공)
    runner_meta = run_result.get("metadata", {})

    summary = {
        "schema_version":   "1.0",
        "run_id":           run_id,
        "spec_id":          spec.get("spec_id", ""),
        "hypothesis_id":    spec.get("hypothesis_id", ""),
        "experiment_version": spec.get("experiment_version", 1),
        "status":           run_result["status"],
        "primary_metric":   {
            "name":   primary_name,
            "value":  primary_value,
            "unit":   spec.get("evaluation_config", {}).get("metric_units", {}).get(primary_name, ""),
            "target": float(target_value),
            "met":    primary_met,
        },
        "secondary_metrics":  secondary,
        "deltas_vs_baseline": {
            "baseline_run_id": previous_run_id,
            "deltas":          deltas,
        },
        "training_stability": {
            "loss_converged": converged,
            "nan_detected":   nan_detected,
            "lr_schedule_ok": True,
            "notes":          "",
        },
        "bottleneck_candidates": [],
        "ablation_findings":     [],
        "confidence":  (
            0.9 if primary_met else
            round(min(0.8, max(0.2, primary_value / float(target_value) * 0.8)), 2)
            if converged and float(target_value) > 0 else
            0.2
        ),
        "recommended_next_actions": [],
        "stderr_tail": run_result.get("stderr_tail", [])[-50:],
        "stdout_tail": (
            run_result.get("stdout_lines", [])[-50:]
            if run_result["status"] == "metrics_parse_error" else []
        ),
        # runner metadata: RunResult.metadata 전체 보존 (필드 유실 방지)
        "runner_metadata": runner_meta,
        # hypothesis implementation audit 결과 로드
        "hypothesis_implementation": _load_hypothesis_audit(pkg_dir),
        "created_at": datetime.now().isoformat(),
    }

    # experiments/{slug}/results/vN/result_summary.json
    slug = slug_from_pkg(pkg_dir)
    ver = version_from_pkg(pkg_dir)
    ver_results_dir = result_version_dir(slug, ver)
    ver_results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ver_results_dir / "result_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # runner_metadata.json 별도 저장
    (ver_results_dir / "runner_metadata.json").write_text(
        json.dumps(summary.get("runner_metadata", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"    [저장] {summary_path}")
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

    print("  [해석 / GPT]    결과 심층 분석...")
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
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
    """Gemini가 짧은 독립 진단 / 2차 의견을 제공한다.

    GPT의 주요 해석과 독립적으로 진단하고, 동의/불일치를 명확히 한다.
    코드를 작성하거나 패치를 제안하지 않는다.
    출력: suggested_path, short_diagnosis, agreement_with_gpt,
           disagreement_reason, main_risk, confidence
    """
    primary = summary["primary_metric"]
    hyp     = hypothesis.get("hypothesis", {})

    prompt = f"""You are a deep learning research second opinion (Gemini role).
Provide a SHORT independent diagnosis. Do NOT write code. Do NOT propose code changes.
Return valid JSON only.

## Hypothesis
{hyp.get('statement_kr', hyp.get('statement', ''))}

## Latest Result
- status: {summary['status']}
- {primary['name']}: {primary['value']:.4f} (target={primary['target']:.2f}, met={primary['met']})
- stability: {json.dumps(summary.get('training_stability', {}), ensure_ascii=False)}

## GPT Interpretation (for reference — you may agree or disagree)
- suggested_path: {gpt_interpretation.get('suggested_path')}
- evidence_strength: {gpt_interpretation.get('evidence_strength')}
- root_cause: {gpt_interpretation.get('root_cause_analysis', '')}
- hypothesis_validity: {gpt_interpretation.get('hypothesis_validity_assessment', '')}

Provide your SHORT independent second opinion:
{{
  "suggested_path": "A|B|C|done",
  "short_diagnosis": "one clear sentence explaining the main issue",
  "agreement_with_gpt": "agree|partial|disagree",
  "disagreement_reason": "if partial or disagree, explain briefly; else empty string",
  "main_risk": "biggest risk if GPT suggestion is followed",
  "confidence": "low|medium|high"
}}"""

    print("  [진단 / Gemini] 2차 의견...")
    try:
        model  = get_gemini_model()
        resp   = model.generate_content(prompt)
        text   = resp.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        result = json.loads(text)
        print(f"    → suggested_path={result.get('suggested_path')}, "
              f"agreement={result.get('agreement_with_gpt')}")
        return result
    except Exception as e:
        print(f"    [경고] Gemini 2차 진단 실패: {e}")
        return {
            "suggested_path": gpt_interpretation.get("suggested_path", "A"),
            "short_diagnosis": f"Gemini call failed: {e}",
            "agreement_with_gpt": "agree",
            "disagreement_reason": "",
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
    gem_agree = gemini_diagnosis.get("agreement_with_gpt", "agree")
    n_runs    = len(run_history)

    # 합의 수준 계산
    if gpt_path == gem_path and gem_agree == "agree":
        agreement_level = "strong"
    elif gpt_path == gem_path or gem_agree in ("agree", "partial"):
        agreement_level = "medium"
    else:
        agreement_level = "weak"

    # Path B 명시적 허용 조건 (모두 충족해야 B 허용)
    # 1. GPT가 B 또는 C를 제안해야 함 (A를 제안하면 B 불가)
    # 2. Gemini가 강한 반대(disagree)가 아니어야 함
    # 3. 합의 수준이 최소 medium이어야 함
    # 4. 실행 횟수 ≥ 2 (smoke/runtime 단일 실패 시 B 불가)
    def _path_b_allowed() -> bool:
        return (
            gpt_path in ("B", "C")
            and gem_agree != "disagree"
            and agreement_level in ("medium", "strong")
            and n_runs >= 2
        )

    # 합의 경로 후보 — 과도한 에스컬레이션 방지
    if primary["met"]:
        consensus_path = "done"
    elif gpt_path == "C" and (n_runs < 3 or gpt_ev != "high" or agreement_level != "strong"):
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

    print("  [패치 / GPT]    Path A 구현 패치 제안...")
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
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
{json.dumps(hypothesis.get('hypothesis', {{}}), ensure_ascii=False, indent=2)}

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
{json.dumps(summary.get('hypothesis_implementation', {{}}), ensure_ascii=False, indent=2)}
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

def _append_results_log(summary: dict, pkg_dir: Path) -> None:
    # experiments/{slug}/results/previous_results.jsonl
    slug = slug_from_pkg(pkg_dir)
    res_dir = results_dir(slug)
    res_dir.mkdir(parents=True, exist_ok=True)
    log_path = res_dir / "previous_results.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"    [append] {log_path}")


# ──────────────────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────────────────

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

        # result_summary 생성 + 기록
        summary = _build_result_summary(pkg, run_result, spec, prev_run_id, prev_metrics)
        _append_results_log(summary, pkg)
        final_summary = summary

        primary = summary["primary_metric"]
        print(f"\n  [결과] {primary['name']} = {primary['value']:.4f} "
              f"(target={primary['target']:.2f}, met={primary['met']})")

        stability = summary.get("training_stability", {})
        run_entry: dict = {
            # 기본 실행 정보
            "run_id":  summary["run_id"],
            "version": version,
            "status":  summary["status"],
            "metrics": summary["primary_metric"],
            # 훈련 안정성 요약
            "training_stability": {
                "loss_converged": stability.get("loss_converged", False),
                "nan_detected":   stability.get("nan_detected", False),
            },
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

        # run_history 마지막 항목에 결정 컨텍스트 보강 (사후검증 결과 + consensus-locked 필드 포함)
        run_history[-1].update({
            "decision_path":         decision.get("path"),
            "candidate_path":        decision.get("candidate_path"),
            "claude_override":       decision.get("claude_override", False),
            "consensus_level":       decision.get("consensus_level", consensus.get("agreement_level")),
            "gpt_suggested_path":    gpt_interp.get("suggested_path"),
            "gemini_suggested_path": gemini_diag.get("suggested_path"),
            "accepted_patch_indexes": decision.get("accepted_patch_indexes",
                                                   decision.get("accepted_gpt_patches", [])),
            "accepted_patch_source": decision.get("accepted_patch_source"),
            "postcheck_override":    decision.get("postcheck_override", False),
            "postcheck_ok":          decision.get("postcheck_ok", True),
            "original_claude_path":  decision.get("original_claude_path"),
            "postcheck_note":        decision.get("postcheck_note"),
        })

        final_decision = decision
        print(f"  [결정] Path {decision['path']} — {decision['justification'][:80]}")

        if decision["path"] == "done":
            print(f"\n  🎉 목표 달성! {primary['name']}={primary['value']:.4f}")
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

        push = subprocess.run(
            ["git", "push", remote, branch],
            capture_output=True, text=True,
        )
        if push.returncode == 0:
            print(f"    [git] 결과 push 완료 → {remote}/{branch}")
        else:
            print(f"    [git 경고] push 실패: {push.stderr.strip()}")
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
