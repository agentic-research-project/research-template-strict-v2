"""
Stage 8: Research Loop — 실험 실행 + Path A/B/C revision 루프

experiments/{slug}_v{N}/ 패키지를 실행하고 결과를 분석한다.
목표 미달이면 Path A (코드 수정) 로 최대 max_rounds회 재시도한다.
Path B/C가 필요하면 revision_request.json을 생성하고 루프를 종료한다.

Path A 변경:
  - generate_experiment_package()에 이전 패키지 + accepted_gpt_patches +
    gemini_review + result_summary를 함께 전달
  - model_generator 내부에서 전체 재생성 대신 최소 diff revision 수행

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

def _gemini_interpret_results(
    pkg_dir: Path,
    summary: dict,
    hypothesis: dict,
    run_history: list[dict],
) -> dict:
    """Gemini가 실험 결과를 해석하고 revision 방향을 제안한다."""
    primary = summary["primary_metric"]
    hyp     = hypothesis.get("hypothesis", {})

    prompt = f"""You are a deep learning research analyst (Gemini role).
Interpret the experiment results and suggest a revision direction.
Return valid JSON only.

## Hypothesis
{hyp.get('statement_kr', hyp.get('statement', ''))}

## Run History ({len(run_history)} runs)
{json.dumps(run_history, ensure_ascii=False, indent=2)}

## Latest Result
- status: {summary['status']}
- {primary['name']}: {primary['value']:.4f} (target={primary['target']:.2f}, met={primary['met']})
- training_stability: {json.dumps(summary.get('training_stability', {}), ensure_ascii=False)}

## Your task
1. Identify the most likely root cause of underperformance
2. Determine if the hypothesis mechanism is fundamentally valid
3. Suggest concrete improvement direction

## Revision criteria
- Path A: implementation issue (code bug, wrong config, unstable training)
- Path B: hypothesis valid but needs scope refinement (50-80% of target met)
- Path C: hypothesis core mechanism contradicted (consistent failure, ≥3 runs)
- done: target met

Return JSON only:
{{
  "suggested_path": "A|B|C|done",
  "evidence_strength": "low|medium|high",
  "root_cause_analysis": "...",
  "bottleneck_candidates": ["..."],
  "improvement_suggestions": ["..."],
  "hypothesis_validity_assessment": "..."
}}"""

    print("  [분석 / Gemini] 결과 해석...")
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
              f"evidence={result.get('evidence_strength')}")
        return result
    except Exception as e:
        print(f"    [경고] Gemini 결과 해석 실패: {e}")
        return {
            "suggested_path": "A",
            "evidence_strength": "low",
            "root_cause_analysis": f"Gemini call failed: {e}",
            "bottleneck_candidates": [],
            "improvement_suggestions": [],
            "hypothesis_validity_assessment": "unknown",
        }


def _gpt_propose_improvements(
    pkg_dir: Path,
    summary: dict,
    gemini_review: dict,
) -> dict:
    """GPT/Codex가 결과 기반으로 코드 개선 패치를 제안한다 (Path A 시)."""
    primary = summary["primary_metric"]

    model_code   = (pkg_dir / "model.py").read_text(encoding="utf-8")   if (pkg_dir / "model.py").exists()   else ""
    default_yaml = (pkg_dir / "configs/default.yaml").read_text(encoding="utf-8") if (pkg_dir / "configs/default.yaml").exists() else ""

    system_msg = (
        "You are a PyTorch optimization engineer (GPT/Codex role). "
        "Propose concrete code patches to improve experiment performance. "
        "Focus on model capacity, training stability, and metric improvement. "
        "Return valid JSON only."
    )

    user_msg = f"""Propose code patches to improve experiment performance.

## Current Result
- {primary['name']}: {primary['value']:.4f} → target: {primary['target']:.2f}
- gap: {primary['target'] - primary['value']:.4f}

## Gemini Analysis
- root_cause: {gemini_review.get('root_cause_analysis', '')}
- bottlenecks: {gemini_review.get('bottleneck_candidates', [])}
- suggestions: {gemini_review.get('improvement_suggestions', [])}

## Current model.py (excerpt)
```python
{model_code[:2500]}
```

## Current default.yaml
```yaml
{default_yaml[:1000]}
```

Propose targeted patches. Each patch must include hypothesis_alignment_check.
Return JSON only:
{{
  "patches": [
    {{
      "target_file": "model.py|configs/default.yaml",
      "rationale": "...",
      "hypothesis_alignment_check": "...",
      "complexity_delta_loc": 5,
      "changes": [
        {{"type": "replace", "old": "exact snippet", "new": "improved snippet"}}
      ]
    }}
  ],
  "expected_improvement": "...",
  "confidence": "low|medium|high"
}}"""

    print("  [분석 / GPT]    개선 패치 제안...")
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
        print(f"    [경고] GPT 개선 제안 실패: {e}")
        return {"patches": [], "expected_improvement": f"GPT call failed: {e}", "confidence": "low"}


def _claude_decide_and_merge(
    pkg_dir: Path,
    summary: dict,
    hypothesis: dict,
    run_history: list[dict],
    gemini_review: dict,
    gpt_proposal: dict,
    max_rounds: int,
) -> dict:
    """
    Claude가 Gemini 리뷰 + GPT 제안을 검토하고
    Path A/B/C/done을 결정한다.
    Path A일 경우 GPT 패치를 적용할 improvement_hints도 생성한다.
    """
    primary  = summary["primary_metric"]
    n_runs   = len(run_history)
    status   = summary["status"]

    # 빠른 결정 — 실행 에러
    if status in ("smoke_failed", "failed", "metrics_parse_error", "timeout"):
        hints = (
            f"status={status}. "
            f"Gemini: {gemini_review.get('root_cause_analysis', '')}. "
            f"GPT patches: {len(gpt_proposal.get('patches', []))}개 제안됨."
        )
        return {"path": "A", "justification": f"실험 실패 ({status})",
                "improvement_hints": hints, "evidence_strength": "low",
                "accepted_gpt_patches": list(range(len(gpt_proposal.get("patches", []))))}

    # 빠른 결정 — 목표 달성
    if primary["met"]:
        return {"path": "done",
                "justification": f"{primary['name']}={primary['value']:.4f} ≥ {primary['target']:.2f}",
                "improvement_hints": "", "evidence_strength": "high",
                "accepted_gpt_patches": []}

    # Claude 최종 판단
    prompt = f"""당신은 연구 파이프라인의 최종 결정자(Claude 역할)입니다.
Gemini의 결과 해석과 GPT의 패치 제안을 검토하고 최종 revision path를 결정하세요.

## 실험 이력 ({n_runs}회)
{json.dumps(run_history, ensure_ascii=False, indent=2)}

## 현재 가설
{json.dumps(hypothesis.get('hypothesis', {}), ensure_ascii=False, indent=2)}

## Gemini 분석
- suggested_path: {gemini_review.get('suggested_path')}
- evidence_strength: {gemini_review.get('evidence_strength')}
- root_cause: {gemini_review.get('root_cause_analysis', '')}
- bottlenecks: {gemini_review.get('bottleneck_candidates', [])}
- suggestions: {gemini_review.get('improvement_suggestions', [])}

## GPT 패치 제안 ({len(gpt_proposal.get('patches', []))}개)
{json.dumps(gpt_proposal.get('patches', []), ensure_ascii=False, indent=2)}

## 결정 규칙 (엄격 적용)
- Path A: 구현 문제 (1~2회 실패, 개선 가능성 있음)
- Path B: 핵심 아이디어 유효, 범위 조정 필요 (target 50~80% 달성)
- Path C: 가설 메커니즘 반증 (≥3회 일관 실패 + Gemini evidence_strength=high)
- done: 목표 달성
에스컬레이션 원칙: A→B→C 순서, 단일 실패는 C 불가

Path A일 경우 improvement_hints에 GPT 패치 중 적용할 항목과 이유를 포함하세요.

아래 JSON으로만 출력:
{{
  "path": "A|B|C|done",
  "justification": "...",
  "improvement_hints": "...",
  "evidence_strength": "low|medium|high",
  "accepted_gpt_patches": [0, 1],
  "rejected_gpt_patches": [2]
}}"""

    try:
        result = parse_json(query_claude(prompt))
        print(f"  [결정 / Claude] path={result.get('path')}, "
              f"evidence={result.get('evidence_strength')}")
        return result
    except Exception as e:
        print(f"    [경고] Claude 최종 결정 실패 ({e}) → 기본값 Path A")
        gap  = primary["target"] - primary["value"]
        path = "A" if n_runs < max_rounds else "B"
        return {
            "path": path,
            "justification": f"{primary['name']}={primary['value']:.4f}, gap={gap:.4f}",
            "improvement_hints": (
                f"Gemini suggests: {gemini_review.get('improvement_suggestions', [])}. "
                f"GPT: {len(gpt_proposal.get('patches', []))}개 패치 제안."
            ),
            "evidence_strength": "low" if n_runs < 2 else "medium",
            "accepted_gpt_patches": list(range(len(gpt_proposal.get("patches", [])))),
        }


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

        run_history.append({
            "run_id":  summary["run_id"],
            "version": version,
            "status":  summary["status"],
            "metrics": summary["primary_metric"],
        })
        prev_run_id  = summary["run_id"]
        prev_metrics = {
            primary["name"]: primary["value"],
            **{m["name"]: m["value"] for m in summary["secondary_metrics"]},
        }

        # ── Multi-model 분석 ───────────────────────────────
        print(f"\n  [분석] Gemini → GPT → Claude (proposal-review-merge)")

        gemini_review = _gemini_interpret_results(pkg, summary, hypothesis, run_history)
        gem_path = _save_proposal(pkg, "gemini_review_result", gemini_review)

        if not primary["met"]:
            gpt_proposal = _gpt_propose_improvements(pkg, summary, gemini_review)
            gpt_path     = _save_proposal(pkg, "gpt_patch_result", gpt_proposal)
        else:
            gpt_proposal = {"patches": [], "confidence": "high"}
            gpt_path     = None

        decision = _claude_decide_and_merge(
            pkg, summary, hypothesis, run_history,
            gemini_review, gpt_proposal, max_rounds,
        )

        # proposal archive
        _archive_proposal(gem_path, decision.get("path") in ("A", "B", "C", "done"))
        if gpt_path:
            accepted_indices = set(decision.get("accepted_gpt_patches", []))
            _archive_proposal(gpt_path, bool(accepted_indices))

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
            all_patches   = gpt_proposal.get("patches", [])
            accepted_idxs = sorted(
                i for i in decision.get("accepted_gpt_patches", [])
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
                gemini_review_ctx     = gemini_review,
                result_summary_ctx    = summary,
            )
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
