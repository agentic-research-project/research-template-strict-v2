"""
Stage 8: Research Loop — 실험 실행 + Path A/B/C revision 루프

experiments/{slug}_v{N}/ 패키지를 실행하고 결과를 분석한다.
목표 미달이면 Path A (코드 수정) 로 최대 max_rounds회 재시도한다.
Path B/C가 필요하면 revision_request.json을 생성하고 루프를 종료한다.

분석 파이프라인 (Multi-model):
  GPT  → 주요 해석자 (심층 결과 해석, suggested_path + evidence)
  Gemini → 독립 2차 진단 (short diagnosis, agreement_with_gpt)
  합의 레이어 → GPT+Gemini 통합, Path C 과도 에스컬레이션 방지
  GPT  → Path A 시 구현 패치 제안 (patch-only, 결정 권한 없음)
  Claude → 최종 결정자 (합의 증거 통합 → path A/B/C/done 결정)

Runner 추상화:
  --runner-type local   (기본값) 로컬 subprocess 실행
  --runner-type gitlab  GitLab CI/CD 트리거 + 결과 수집

사용법:
  python -m lab.research_loop \\
    --pkg-dir       experiments/{slug}_v1 \\
    --topic-file    reports/{slug}/topic_analysis.json \\
    --hypothesis-file reports/{slug}/hypothesis.json \\
    --code-file     reports/{slug}/code_analysis.json \\
    [--max-rounds 3] [--runner-type local]
"""

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

from lab.config import query_claude, parse_json, get_openai_client, get_gemini_model
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
                "--runner-type local 로 변경하거나 "
                "runner를 완전히 구현한 후 재시도하세요."
            ),
        }
        print("\n[FATAL] Runner 준비 안 됨 — 실험 루프 진입 중단")
        print(json.dumps(error_info, ensure_ascii=False, indent=2))
        sys.exit(1)


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
            "unit":   "dB" if primary_name == "psnr" else "",
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
        "confidence":  0.5 if converged else 0.2,
        "recommended_next_actions": [],
        "stderr_tail": run_result.get("stderr_tail", [])[-50:],
        "stdout_tail": (
            run_result.get("stdout_lines", [])[-50:]
            if run_result["status"] == "metrics_parse_error" else []
        ),
        # runner metadata: local 또는 gitlab 실행 정보
        "runner_metadata": {
            "runner":       runner_meta.get("runner", "local"),
            "job_id":       runner_meta.get("job_id", ""),
            "duration_s":   runner_meta.get("duration_s", 0),
            "artifact_uri": runner_meta.get("artifact_uri", ""),
            "job_url":      runner_meta.get("job_url", ""),
            "git_sha":      runner_meta.get("git_sha", ""),
        },
        "created_at": datetime.now().isoformat(),
    }

    summary_path = pkg_dir / "result_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
Key mechanism: {hyp.get('key_mechanism', hyp.get('mechanism', ''))}

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

## Revision criteria (strict)
- Path A: implementation issue — hypothesis still valid (code bug, config, training instability)
- Path B: hypothesis directionally valid but needs refinement (50-80% target met, scope too broad)
- Path C: hypothesis CORE MECHANISM contradicted by repeated strong evidence (requires ≥3 runs)
- done: target metric achieved

Rules: single failure cannot go to Path C. Escalation must follow A → B → C.

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
            model="gpt-4o",
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
            model="gpt-4o",
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
) -> dict:
    """Claude가 GPT 해석 + Gemini 진단 + 합의 요약 + GPT 패치를 종합하여 최종 결정한다.

    Claude는 독자적 재해석 대신 제공된 증거를 통합하여 결정한다.
    Path B/C는 높은 기준을 요구하며, 합의 수준을 존중한다.
    출력: path, decision_reason, consensus_level, accepted_patch_indexes,
           rejected_patch_indexes, improvement_hints, evidence_strength
    """
    primary  = summary["primary_metric"]
    n_runs   = len(run_history)
    status   = summary["status"]
    patches  = gpt_patches.get("patches", [])

    # 빠른 결정 — 목표 달성
    if primary["met"]:
        return {
            "path": "done",
            "decision_reason": f"{primary['name']}={primary['value']:.4f} ≥ {primary['target']:.2f}",
            "justification":   f"{primary['name']}={primary['value']:.4f} ≥ {primary['target']:.2f}",
            "consensus_level": "strong",
            "improvement_hints": "",
            "evidence_strength": "high",
            "accepted_patch_indexes": [],
            "rejected_patch_indexes": [],
            "accepted_gpt_patches":   [],
        }

    # 빠른 결정 — 실행 에러 (Path A 기본)
    if status in ("smoke_failed", "failed", "metrics_parse_error", "timeout"):
        accepted = list(range(len(patches)))
        hints = (
            f"status={status}. GPT root_cause: {gpt_interpretation.get('root_cause_analysis', '')}. "
            f"Applying all {len(patches)} GPT patches."
        )
        return {
            "path": "A",
            "decision_reason": f"실험 실패 ({status}) → 구현 수정 필요",
            "justification":   f"실험 실패 ({status})",
            "consensus_level": consensus.get("agreement_level", "weak"),
            "improvement_hints": hints,
            "evidence_strength": "low",
            "accepted_patch_indexes": accepted,
            "rejected_patch_indexes": [],
            "accepted_gpt_patches":   accepted,
        }

    # Claude 최종 판단 (합의 증거 통합)
    prompt = f"""당신은 연구 파이프라인의 최종 결정자(Claude 역할)입니다.
아래 다중 모델 분석 결과를 통합하여 최종 revision path를 결정하세요.
독자적 재해석 대신 제공된 증거를 존중하고 통합하세요.

## 실험 이력 ({n_runs}회)
{json.dumps(run_history, ensure_ascii=False, indent=2)}

## 현재 가설
{json.dumps(hypothesis.get('hypothesis', {}), ensure_ascii=False, indent=2)}

## GPT 심층 해석 (주요 해석자)
- suggested_path: {gpt_interpretation.get('suggested_path')}
- evidence_strength: {gpt_interpretation.get('evidence_strength')}
- confidence: {gpt_interpretation.get('confidence')}
- root_cause_analysis: {gpt_interpretation.get('root_cause_analysis', '')}
- hypothesis_validity_assessment: {gpt_interpretation.get('hypothesis_validity_assessment', '')}
- bottleneck_candidates: {gpt_interpretation.get('bottleneck_candidates', [])}
- why_not_other_paths: {gpt_interpretation.get('why_not_other_paths', '')}

## Gemini 2차 진단 (독립 의견)
- suggested_path: {gemini_diagnosis.get('suggested_path')}
- short_diagnosis: {gemini_diagnosis.get('short_diagnosis', '')}
- agreement_with_gpt: {gemini_diagnosis.get('agreement_with_gpt')}
- disagreement_reason: {gemini_diagnosis.get('disagreement_reason', '')}
- main_risk: {gemini_diagnosis.get('main_risk', '')}

## 합의 요약
- consensus_path_candidate: {consensus.get('consensus_path_candidate')}
- agreement_level: {consensus.get('agreement_level')}
- escalation_risk: {consensus.get('escalation_risk')}
- major_disagreements: {consensus.get('major_disagreements', [])}
- notes_for_claude: {consensus.get('notes_for_claude', '')}

## GPT 패치 제안 ({len(patches)}개 — Path A 시에만 적용)
{json.dumps(patches, ensure_ascii=False, indent=2)}

## 결정 규칙 (엄격 적용)

Path A = 구현/훈련/실험설계 문제 (가설 자체는 유효)
  - 적용 조건: 약한 합의도 허용, 단일 실패도 가능
  - 예: 코드 버그, 학습 불안정, 설정 오류, NaN, smoke 실패

Path B = 가설 방향성 유효하나 범위/조건/클레임 정제 필요
  - 적용 조건 (모두 충족 필요):
    * GPT가 B 또는 C를 제안한 경우
    * Gemini가 강한 반대(disagree) 아닌 경우
    * consensus agreement_level이 medium 이상인 경우
    * 실행 횟수 ≥ 2 (smoke/runtime 단일 실패 시 B 불가)
  - 단순 구현 실패, smoke 실패, 단일 실행 실패는 B 불가
  - consensus_path_candidate가 A이면 B 선택을 강한 이유로만 허용

Path C = 가설 핵심 메커니즘 또는 주장하는 이점이 반증됨
  - 적용 조건 (모두 충족 필수):
    * 강한 합의 (agreement_level=strong)
    * ≥3회 실행 증거
    * evidence_strength=high
    * escalation_risk=high (blocked이면 C 불가)

done = 목표 달성

에스컬레이션 원칙: A→B→C 순서, 단일 실패는 B 이상 불가, escalation_risk=blocked이면 C 불가.
consensus_path_candidate를 존중하되, B/C 선택 시 위 조건을 명시적으로 확인하세요.

Path A 결정 시: accepted_patch_indexes에 적용할 패치 인덱스, improvement_hints에 이유를 포함하세요.
Path B 결정 시: 위 조건 충족 여부를 decision_reason에 명시하세요.
Path C 결정 시: 위 조건 충족 여부를 decision_reason에 명시하세요.

아래 JSON으로만 출력:
{{
  "path": "A|B|C|done",
  "decision_reason": "...",
  "consensus_level": "weak|medium|strong",
  "improvement_hints": "...",
  "evidence_strength": "low|medium|high",
  "accepted_patch_indexes": [0, 1],
  "rejected_patch_indexes": [2]
}}"""

    try:
        result = parse_json(query_claude(prompt))
        # 하위 호환: accepted_gpt_patches 필드 동기화
        result["accepted_gpt_patches"]   = result.get("accepted_patch_indexes", [])
        result["rejected_gpt_patches"]   = result.get("rejected_patch_indexes", [])
        result.setdefault("justification", result.get("decision_reason", ""))
        print(f"  [결정 / Claude] path={result.get('path')}, "
              f"consensus={result.get('consensus_level')}, "
              f"evidence={result.get('evidence_strength')}")
        return result
    except Exception as e:
        print(f"    [경고] Claude 최종 결정 실패 ({e}) → 합의 경로 기본 사용")
        gap          = primary["target"] - primary["value"]
        fallback_path = consensus.get("consensus_path_candidate", "A")
        accepted      = list(range(len(patches)))
        return {
            "path": fallback_path,
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

    Claude 결정이 guardrail을 위반한 경우 자동으로 다운그레이드하고
    postcheck_override=True와 postcheck_note에 사유를 기록한다.

    반환 필드:
      path, decision_reason, consensus_level,
      postcheck_applied, postcheck_override, postcheck_note,
      original_claude_path (override 시만 추가)
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
            return False, f"insufficient runs (n_runs={n_runs}, need ≥2)"
        if status in ("smoke_failed", "failed", "metrics_parse_error", "timeout") and n_runs < 2:
            return False, f"pure execution failure without prior success (status={status})"
        return True, ""

    def _c_allowed() -> tuple[bool, str]:
        """Path C 허용 조건. (허용여부, 실패 사유) 반환."""
        if n_runs < 3:
            return False, f"insufficient runs (n_runs={n_runs}, need ≥3)"
        if agreement_lvl != "strong":
            return False, f"consensus not strong ({agreement_lvl})"
        if gpt_ev != "high":
            return False, f"GPT evidence_strength not high ({gpt_ev})"
        if consensus.get("escalation_risk") == "blocked":
            return False, "escalation_risk=blocked in consensus"
        # 구현/훈련 불안정 케이스는 C 불가
        stab = summary.get("training_stability", {})
        if stab.get("nan_detected") or not stab.get("loss_converged", True):
            return False, "training instability detected — implementation issue, not hypothesis failure"
        return True, ""

    # ── 사후 검증 로직 ───────────────────────────────────────

    override      = False
    override_note = None
    original_path = path

    if path == "done":
        # done 조건 재확인: primary_met=True여야 함
        if not primary["met"]:
            corrected = consensus_path if consensus_path in ("A", "B") else "A"
            override, override_note = True, (
                f"Claude returned done but primary metric not met "
                f"({primary['name']}={primary['value']:.4f} < {primary['target']:.2f}) "
                f"— corrected to {corrected}"
            )
            path = corrected

    elif path == "C":
        ok, reason = _c_allowed()
        if not ok:
            # C → B 시도, B도 안 되면 A
            b_ok, _ = _b_allowed()
            corrected = "B" if b_ok else "A"
            override, override_note = True, (
                f"Path C not allowed: {reason} — downgraded to {corrected}"
            )
            path = corrected

    elif path == "B":
        ok, reason = _b_allowed()
        if not ok:
            corrected = consensus_path if consensus_path == "A" else "A"
            override, override_note = True, (
                f"Path B not allowed: {reason} — downgraded to {corrected}"
            )
            path = corrected

    # ── 결과 조립 ─────────────────────────────────────────────

    result = dict(decision)   # 원본 필드 모두 보존
    result["path"]              = path
    result["postcheck_applied"] = True
    result["postcheck_override"] = override
    result["postcheck_note"]    = override_note

    if override:
        result["original_claude_path"] = original_path
        # justification / decision_reason 업데이트
        note_prefix = f"[postcheck override: {original_path}→{path}] "
        result["decision_reason"] = note_prefix + result.get("decision_reason", "")
        result["justification"]   = note_prefix + result.get("justification", "")
        print(f"    [사후검증 ⚠]  Claude={original_path} → 수정={path}: {override_note}")
    else:
        result["original_claude_path"] = original_path
        print(f"    [사후검증 ✓]  Claude={original_path} 유지")

    return result


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

def _append_results_log(summary: dict) -> None:
    log_path = Path("results/previous_results.jsonl")
    log_path.parent.mkdir(exist_ok=True)
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
    runner_type: str = "local",
    runner_config: dict | None = None,
) -> dict:
    """실험 실행 + Path A/B/C revision 루프. 최종 result_summary 반환."""
    topic       = json.loads(Path(topic_file).read_text(encoding="utf-8"))
    hypothesis  = json.loads(Path(hypothesis_file).read_text(encoding="utf-8"))
    reports_dir = Path(topic_file).parent

    pkg     = Path(pkg_dir)
    m       = re.search(r"_v(\d+)$", pkg.name)
    version = int(m.group(1)) if m else 1

    # runner 초기화 + 준비 상태 확인 (미구현이면 즉시 종료)
    runner: BaseRunner = create_runner(runner_type, runner_config)
    _check_runner_ready(runner)
    print(f"  [Runner] {runner.__class__.__name__} 사용")

    run_history:   list[dict] = []
    prev_run_id:   str | None = None
    prev_metrics:  dict       = {}
    final_summary: dict       = {}

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
            spec = {
                "schema_version": "1.0", "spec_id": f"{slug}_v{version}",
                "hypothesis_id": str(uuid.uuid4())[:8],
                "experiment_version": version, "topic_slug": slug,
                "evaluation_config": {
                    "primary_metric": "psnr", "target_value": 30.0,
                    "secondary_metrics": [], "test_set": "val_split",
                },
            }

        # result_summary 생성 + 기록
        summary = _build_result_summary(pkg, run_result, spec, prev_run_id, prev_metrics)
        _append_results_log(summary)
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
        # 파이프라인: GPT(해석) → Gemini(진단) → 합의 → GPT(패치, Path A 한정) → Claude(결정)
        print(f"\n  [분석] GPT(해석) → Gemini(진단) → 합의 → GPT(패치) → Claude(결정)")

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

        # 5. Claude: 최종 결정자
        decision = _claude_final_decision(
            pkg, summary, hypothesis, run_history,
            gpt_interp, gemini_diag, consensus, gpt_patches, max_rounds,
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

        # run_history 마지막 항목에 결정 컨텍스트 보강 (사후검증 결과 포함)
        run_history[-1].update({
            "decision_path":         decision.get("path"),
            "consensus_level":       decision.get("consensus_level", consensus.get("agreement_level")),
            "gpt_suggested_path":    gpt_interp.get("suggested_path"),
            "gemini_suggested_path": gemini_diag.get("suggested_path"),
            "accepted_patch_indexes": decision.get("accepted_patch_indexes",
                                                   decision.get("accepted_gpt_patches", [])),
            "postcheck_override":    decision.get("postcheck_override", False),
            "original_claude_path":  decision.get("original_claude_path"),
            "postcheck_note":        decision.get("postcheck_note"),
        })

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

    return final_summary


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Loop — 실험 실행 + revision")
    parser.add_argument("--pkg-dir",         required=True,
                        help="experiments/{slug}_v{N} 패키지 디렉토리")
    parser.add_argument("--topic-file",      required=True)
    parser.add_argument("--hypothesis-file", required=True)
    parser.add_argument("--code-file",       required=True)
    parser.add_argument("--max-rounds",      type=int, default=3)
    parser.add_argument("--config-file",     default="configs/default.yaml")
    # Runner 선택
    parser.add_argument("--runner-type",     default="local", choices=["local", "gitlab"],
                        help="실험 실행 방식 (기본값: local)")
    # GitLab runner 설정 (--runner-type gitlab 시 사용)
    parser.add_argument("--gitlab-url",         default=os.getenv("GITLAB_URL", ""))
    parser.add_argument("--gitlab-token",       default=os.getenv("GITLAB_TOKEN", ""))
    parser.add_argument("--gitlab-project-id",  default=os.getenv("GITLAB_PROJECT_ID", ""))
    parser.add_argument("--gitlab-ref",         default=os.getenv("GITLAB_REF", "main"))
    args = parser.parse_args()

    runner_config = {
        "gitlab_url":        args.gitlab_url,
        "gitlab_token":      args.gitlab_token,
        "gitlab_project_id": args.gitlab_project_id,
        "gitlab_ref":        args.gitlab_ref,
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
