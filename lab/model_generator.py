"""
Stage 7: PyTorch Fabric 실험 패키지 생성 (Multi-model Proposal-Review-Merge)

흐름:
  1. Claude    → 기반 코드 생성 (model.py / module.py / data.py / default.yaml)
  2. GPT/Codex → 코드 패치 제안 (proposals/gpt_patch_*.json)
  3-A. Gemini  → 설계 리뷰     (proposals/gemini_review_*.json)
  3-B. Gemini  → 패치별 투표   (proposals/gemini_patch_ballot_*.json)
  3-C. Claude  → 패치별 독립 판정 (proposals/claude_patch_ballot_*.json)
  3-D. 2-of-3  → 패치별 합의 결정 (proposals/merge_decision_*.json)
  4. Claude    → 합의 결과 집행 + 최종 파일 확정 (reject 패치 제외)
  5. Validation → syntax + smoke + forward 검증, 실패 시 1회 repair

Path A revision 모드 (revised_from + revision_path="A"):
  - 이전 패키지를 그대로 복사 후 최소 diff 적용
  - 전체 재생성 대신 변경이 필요한 파일만 수정

구조:
  experiments/{slug}/runs/v{N}/
  ├── train.py            (template 복사 — 수정 금지)
  ├── model.py            (Claude 생성 + merge)
  ├── module.py           (Claude 생성 + merge)
  ├── data.py             (Claude 생성 + merge)
  ├── configs/default.yaml(Claude 생성 + merge)
  ├── configs/fast.yaml   (template 복사)
  ├── scripts/smoke_test.py (template 복사)
  ├── tests/test_forward.py (template 복사)
  ├── proposals/          (GPT·Gemini 제안 아카이브)
  └── artifacts/

사용법:
  python -m lab.model_generator \\
    --topic-file      experiments/{slug}/reports/topic_analysis.json \\
    --hypothesis-file experiments/{slug}/reports/hypothesis.json \\
    --code-file       experiments/{slug}/reports/code_analysis.json \\
    [--version 1] [--revised-from experiments/{slug}/runs/v1]
    [--revision-path A] [--improvement-hints "..."]
"""

import argparse
import json
import py_compile
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from lab.config import query_claude, parse_json, get_openai_client, get_gemini_model, OPENAI_MODEL, topic_slug as _topic_slug

TEMPLATE_DIR = Path(__file__).parent.parent / "experiments" / "template"
SCHEMAS_DIR  = Path(__file__).parent.parent / "schemas"


# ──────────────────────────────────────────────────────────
# 제안 파일 저장 헬퍼
# ──────────────────────────────────────────────────────────

def _save_proposal(pkg_dir: Path, prefix: str, data: dict) -> Path:
    """proposals/{prefix}_{timestamp}.json 으로 저장하고 경로 반환."""
    proposals_dir = pkg_dir / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)  # parents=True: pkg_dir가 삭제된 경우 재생성
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = proposals_dir / f"{prefix}_{ts}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [proposals] {path.name} 저장")
    return path


def _archive_proposal(path: Path, accepted: bool) -> None:
    """병합 완료 후 _accepted / _rejected suffix 추가."""
    suffix = "_accepted" if accepted else "_rejected"
    new_name = path.stem + suffix + path.suffix
    path.rename(path.parent / new_name)


# ──────────────────────────────────────────────────────────
# Task-family engine helpers
# ──────────────────────────────────────────────────────────

def _scan_data_tree(data_path: str, max_depth: int = 3, max_files: int = 30) -> str:
    """데이터 경로의 디렉토리 구조를 스캔하여 텍스트 트리로 반환한다.

    파일 내용은 읽지 않음 (보안). 이름/확장자/개수만 수집.
    """
    from pathlib import Path
    import os

    if not data_path or not data_path.strip():
        return "(데이터 경로 미지정 — 공개 데이터셋 자동 다운로드 필요)"
    root = Path(data_path)
    if not root.exists():
        return f"(경로 없음: {data_path} — 공개 데이터셋이면 자동 다운로드 구현 필요)"
    if not root.is_dir():
        return f"(디렉토리 아님: {data_path})"

    lines: list[str] = []
    file_count = 0
    ext_counts: dict[str, int] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames.clear()
            continue

        indent = "  " * depth
        rel = Path(dirpath).relative_to(root)
        dir_label = str(rel) if str(rel) != "." else root.name
        n_files = len(filenames)
        lines.append(f"{indent}{dir_label}/ ({n_files} files)")

        for fname in filenames[:5]:  # 디렉토리당 최대 5개 파일명 표시
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "no_ext"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            if file_count < max_files:
                lines.append(f"{indent}  {fname}")
            file_count += 1

        remaining = n_files - min(5, n_files)
        if remaining > 0:
            lines.append(f"{indent}  ... +{remaining} more files")
            for fname in filenames[5:]:
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else "no_ext"
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
            file_count += remaining

    # 요약
    lines.append(f"\nSummary: {file_count} total files")
    lines.append(f"Extensions: {', '.join(f'{ext}({cnt})' for ext, cnt in sorted(ext_counts.items(), key=lambda x: -x[1]))}")

    return "\n".join(lines)


def _flatten_task_bundle(inp: dict, hyp: dict) -> dict:
    """task_family_bundle을 experiment_spec top-level에 펼친다."""
    bundle = _get_task_bundle(inp, hyp)
    prior = bundle.get("generation_prior", {})

    # 최신 트렌드 참조 (코드 생성 시 반영)
    from lab.task_families import LATEST_TRENDS
    trend_hints = {
        "modern_optimizers": LATEST_TRENDS.get("training_techniques", [])[:3],
        "modern_augmentation": LATEST_TRENDS.get("augmentation", [])[:2],
        "modern_architectures": LATEST_TRENDS.get("architectures", [])[:3],
        "modern_losses": LATEST_TRENDS.get("losses", [])[:3],
    }

    return {
        "task_family": bundle["task_family"],
        "pattern_candidates": bundle.get("pattern_candidates", []),
        "family_contract": bundle.get("family_contract", {}),
        "generation_prior": prior,
        "literature_code_prior": bundle.get("literature_code_prior", {}),
        "contract_tests": bundle.get("contract_tests", []),
        "starter_skeleton_path": bundle.get("skeleton_path", ""),
        "must_not_do": prior.get("must_not_do", []),
        "synthesized_baselines": bundle.get("synthesized_baselines", []),
        "latest_trends": trend_hints,
    }


def _get_task_bundle(inp: dict, hyp: dict) -> dict:
    """topic/hypothesis에서 task_family를 추론하고 bundle을 반환한다."""
    from lab.task_families import infer_task_family, get_task_family_bundle
    family = infer_task_family(
        inp.get("topic", ""),
        inp.get("target_metric", ""),
        inp.get("problem_definition", ""),
    )
    return get_task_family_bundle(family)


def _merge_baselines(code_baseline_str: str, inp: dict, hyp: dict) -> list[dict]:
    """code_analysis baseline + task-family synthesized baseline을 병합한다."""
    from lab.task_families import infer_task_family, synthesize_baselines
    # 1. code_analysis에서 추출
    lit_baselines = [
        {"name": b.strip(), "source": "literature"}
        for b in code_baseline_str.split(",") if b.strip()
    ]
    # 2. task-family 내부 prior에서 추가
    family = infer_task_family(
        inp.get("topic", ""),
        inp.get("target_metric", ""),
        inp.get("problem_definition", ""),
    )
    synth = synthesize_baselines(family)
    # 중복 제거
    existing_names = {b["name"] for b in lit_baselines}
    for sb in synth:
        if sb["name"] not in existing_names:
            lit_baselines.append(sb)
    return lit_baselines


# ──────────────────────────────────────────────────────────
# experiment_spec.json 생성
# ──────────────────────────────────────────────────────────

def _build_experiment_spec(
    topic: dict,
    hypothesis: dict,
    code_analysis: dict,
    version: int,
    revised_from: str | None,
    revision_path: str | None,
) -> dict:
    inp      = topic.get("input", {})
    hyp      = hypothesis.get("hypothesis", {})
    exp_plan = hypothesis.get("experiment_plan", {})
    slug     = _topic_slug(inp.get("topic", "research"))
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 지표 추출 (도메인 무관 일반화) ────────────────────────────────
    metrics_list = exp_plan.get("evaluation_metrics", [])
    primary_name: str = ""
    secondary:    list[str] = []
    metric_units: dict[str, str] = {}

    for i, m in enumerate(metrics_list):
        # 첫 번째 토큰을 지표명으로, 괄호 내 단위 추출 (e.g., "PSNR (dB)", "Accuracy (%)")
        key = re.split(r"[\s\(\-—/]", m.strip())[0].lower().replace("-", "_").strip("_")
        if not key:
            continue
        unit_match = re.search(r"\(([^)]+)\)", m)
        metric_units[key] = unit_match.group(1) if unit_match else ""
        if i == 0:
            primary_name = key
        else:
            secondary.append(key)

    # target_metric에서 첫 번째 지표명을 primary로 대체 (더 명시적)
    target_raw   = inp.get("target_metric", "")
    if not primary_name and target_raw:
        first_tok = re.split(r"[\s,\(\-]", target_raw.strip())[0].lower().replace("-", "_").strip("_")
        if first_tok:
            primary_name = first_tok

    # 최후 fallback
    if not primary_name:
        primary_name = "primary_metric"

    # target_val: 숫자 추출 (단위 무관)
    target_val = 0.0
    m_val = re.search(r"(\d+(?:\.\d+)?)", target_raw)
    if m_val:
        target_val = float(m_val.group(1))

    # param budget: constraints에서 추출
    param_budget = 5.0
    m_param = re.search(r"(\d+(?:\.\d+)?)\s*[Mm]", inp.get("constraints", ""))
    if m_param:
        param_budget = float(m_param.group(1))

    # training_config: experiment_plan에서 우선 참조
    train_plan = exp_plan.get("training", exp_plan.get("training_config", {}))

    return {
        "schema_version": "1.0",
        "spec_id":         f"{slug}_v{version}_{ts}",
        "hypothesis_id":   hypothesis.get("hypothesis_id", str(uuid.uuid4())[:8]),
        "experiment_version": version,
        "topic_slug":      slug,
        "created_at":      datetime.now().isoformat(),
        "revised_from_spec_id": revised_from,
        "revision_path":   revision_path,
        "model_architecture": {
            "name":           hyp.get("title", exp_plan.get("architecture", "CustomNet")),
            "description":    hyp.get("statement_kr", hyp.get("statement", "")),
            "key_components": exp_plan.get("key_components", []),
            "param_budget_M": param_budget,
        },
        "training_config": {
            "epochs":        train_plan.get("epochs", 50),
            "batch_size":    train_plan.get("batch_size", 16),
            "optimizer":     train_plan.get("optimizer", "adamw"),
            "lr":            float(train_plan.get("lr", train_plan.get("learning_rate", 1e-4))),
            "loss_function": train_plan.get("loss_function", "auto"),
            "seed":          train_plan.get("seed", 42),
            "precision":     train_plan.get("precision", "bf16-mixed"),
            "gradient_clip": train_plan.get("gradient_clip", 1.0),
        },
        "evaluation_config": {
            "primary_metric":    primary_name,
            "target_value":      target_val,
            "secondary_metrics": secondary,
            "metric_units":      metric_units,
            "test_set":          exp_plan.get("test_set", "val_split"),
            "eval_frequency":    exp_plan.get("eval_frequency", 1),
        },
        # 가설 구현 계약 — mechanism / target_metric / constraints 추적
        "hypothesis_contract": {
            "mechanism": (
                hyp.get("expected_mechanism")
                or hyp.get("key_mechanism")
                or hyp.get("mechanism", "")
            ),
            "target_metric_raw": inp.get("target_metric", ""),
            "constraints_raw": inp.get("constraints", ""),
            "architecture_hint": (
                hyp.get("architecture")
                or exp_plan.get("architecture", "")
            ),
        },
        "constraints_structured": inp.get("constraints_structured", {}),
        "data_config": {
            "data_path": inp.get("data_path", ""),
            "dataset_name": exp_plan.get("dataset", hyp.get("dataset", "")),
        },
        # Task-family engine — top-level 승격 (generation prompt가 바로 참조)
        **_flatten_task_bundle(inp, hyp),
        "ablations": [],
        "baselines": _merge_baselines(
            code_analysis.get("recommended_baseline", ""),
            inp, hyp,
        ),
        "output_contract": {
            "stdout_pattern": "^METRICS:\\{.*\\}$",
            "required_keys":  [primary_name] + secondary,
        },
    }


# ──────────────────────────────────────────────────────────
# Step 1 — Claude: 기반 코드 생성 (template 기반, 최초 생성)
# ──────────────────────────────────────────────────────────

def _claude_generate_base(
    topic: dict,
    hypothesis: dict,
    code_analysis: dict,
    spec: dict,
    improvement_hints: str = "",
) -> dict:
    """Claude가 model.py / module.py / data.py / default.yaml 초안을 생성한다."""
    inp      = topic.get("input", {})
    hyp      = hypothesis.get("hypothesis", {})
    exp_plan = hypothesis.get("experiment_plan", {})
    components = code_analysis.get("reusable_components", [])[:4]
    tips       = code_analysis.get("implementation_tips", [])[:5]

    component_info = "\n".join(
        f"- [{c['type']}] {c['name']}: {c['description']}"
        for c in components
    )
    primary      = spec["evaluation_config"]["primary_metric"]
    secondary    = spec["evaluation_config"]["secondary_metrics"]
    param_budget = spec["model_architecture"]["param_budget_M"]

    improvement_section = (
        f"\n## 이전 실험 개선 요청\n{improvement_hints}" if improvement_hints else ""
    )

    arch      = spec["model_architecture"]
    tr_cfg    = spec["training_config"]
    ev_cfg    = spec["evaluation_config"]
    out_ctr   = spec.get("output_contract", {})
    baselines = spec.get("baselines", [])

    # family-first 정보 추출
    task_family = spec.get("task_family", "classification")
    family_contract = spec.get("family_contract", {})
    must_not_do = spec.get("must_not_do", [])
    gen_prior = spec.get("generation_prior", {})
    lit_prior = spec.get("literature_code_prior", {})
    pattern_cands = spec.get("pattern_candidates", [])
    skeleton_path = spec.get("starter_skeleton_path", "")

    family_section = f"""## [1순위] Task Family Contract — 이 코드는 {task_family} 패키지다
- task_family: {task_family}
- family_contract: {json.dumps(family_contract, ensure_ascii=False)[:500]}
- must_not_do (어기면 실패): {must_not_do}
- pattern_candidates: {[p.get('pattern_id','') for p in pattern_cands[:3]]}
- starter_skeleton: {skeleton_path}
- generation_prior.critical_interfaces: {gen_prior.get('critical_interfaces', [])}
- generation_prior.likely_failure_modes: {gen_prior.get('likely_failure_modes', [])}
- literature_hints: arch={lit_prior.get('architecture_hint','')}, loss={lit_prior.get('loss_hint','')}, eval={lit_prior.get('evaluation_hint','')}

⚠️ family_contract를 어기면 validation gate에서 차단됩니다.
⚠️ must_not_do를 어기면 실패로 판정됩니다.
⚠️ skeleton이 있으면 skeleton을 기반으로 patch하세요 (from-scratch보다 안정적).

## [참고] Latest Trends (2024-2025 — 적용 가능하면 반영)
- optimizers: {spec.get('latest_trends', {}).get('modern_optimizers', [])}
- augmentation: {spec.get('latest_trends', {}).get('modern_augmentation', [])}
- architectures: {spec.get('latest_trends', {}).get('modern_architectures', [])}
- losses: {spec.get('latest_trends', {}).get('modern_losses', [])}
⚠️ 최신 기법은 가설과 constraints에 부합할 때만 적용. 무조건 적용 금지.
"""

    prompt = f"""당신은 PyTorch Fabric 전문 딥러닝 엔지니어이며, 이 실험 패키지의 **유일한 코드 작성자**입니다.
GPT/Codex는 이후 패치만 제안하고, Gemini는 설계만 리뷰하며, 최종 merge도 당신이 수행합니다.
코드 생성 결정은 아래 task family contract → experiment_spec 순서로 따르세요.

{family_section}
## [2순위] 구현 계약 (experiment_spec)
- spec_id: {spec['spec_id']}
- model_architecture:
    name: {arch['name']}
    key_components: {arch.get('key_components', [])}
    param_budget_M: ≤ {param_budget}M
- training_config:
    optimizer: {tr_cfg['optimizer']}, lr: {tr_cfg['lr']}, epochs: {tr_cfg['epochs']}
    loss_function: {tr_cfg['loss_function']}, batch_size: {tr_cfg['batch_size']}, seed: {tr_cfg['seed']}
    precision: {tr_cfg.get('precision','bf16-mixed')}, gradient_clip: {tr_cfg.get('gradient_clip',1.0)}
- evaluation_config:
    primary_metric: {primary} ≥ {ev_cfg['target_value']}
    secondary_metrics: {secondary}
- output_contract:
    stdout_pattern: {out_ctr.get('stdout_pattern', 'METRICS:{{...}}')}
    required_keys: {out_ctr.get('required_keys', [primary])}
- baselines: {[b['name'] for b in baselines]}

## [가설 구현 계약] hypothesis_contract (반드시 코드에 구현)
- mechanism: {spec.get('hypothesis_contract', {}).get('mechanism', '')}
  → 반드시 아키텍처 요소, 학습 요소, 데이터 처리 요소 중 하나 이상으로 구체화할 것
  → 자연어 설명에만 있고 코드에 없는 mechanism은 audit 실패 대상
- target_metric_raw: {spec.get('hypothesis_contract', {}).get('target_metric_raw', '')}
  → validation 계산 및 stdout METRICS 키로 직접 연결할 것
- constraints_raw: {spec.get('hypothesis_contract', {}).get('constraints_raw', '')}
  → default.yaml 및 모델 크기/훈련 설정에 반영할 것
- architecture_hint: {spec.get('hypothesis_contract', {}).get('architecture_hint', '')}

## [데이터] 데이터 경로 + 구조
- data_path: {spec.get('data_config', {}).get('data_path', '')}
- dataset_name: {spec.get('data_config', {}).get('dataset_name', '')}
- 데이터 디렉토리 구조:
{_scan_data_tree(spec.get('data_config', {}).get('data_path', ''))}

  → data_path가 명시되어 있으면 **반드시 해당 경로의 실제 데이터를 로드**해야 함 (합성 데이터 생성 금지)
  → data.py는 위 디렉토리 구조에 맞게 데이터를 로드해야 함
  → 표준 ImageFolder 구조가 아닐 수 있음 — 파일 확장자, 폴더 구조, 라벨 파일 유무를 반드시 확인
  → data_path가 비어있거나 존재하지 않으면 공개 코드 및 논문에 언급된 공개 데이터셋을 다운로드 받아 구현 (torchvision, 해당 데이터 url 등)
  → 데이터 파일 내용을 읽거나 API로 전송하지 마라 (보안)

## [과학적 근거] 연구 가설
- 주제: {inp.get('topic', '')}
- 가설: {hyp.get('statement_kr', hyp.get('statement', ''))}
- 핵심 아키텍처: {exp_plan.get('architecture', '')}
- 제약: {inp.get('constraints', '')}

## [참고] code_analysis 컴포넌트
{component_info if component_info else '없음'}

## [참고] 구현 팁
{chr(10).join(f'- {t}' for t in tips)}
{improvement_section}

## 생성 규칙 (spec 우선)
⚠️ 아래 함수명/클래스명은 train.py가 호출하므로 **절대 변경 금지**:
1. model.py: `build_model(config) -> nn.Module` — 반드시 이 함수명 사용
2. module.py: `TrainingModule(model, config)` — 반드시 이 클래스명 사용, train_epoch/val_epoch 메서드 필수
3. data.py: `build_dataloaders(config)` — 반드시 이 함수명 사용 (create_dataloaders, get_dataloaders 등 금지)
4. default.yaml: **반드시 flat 구조** (nested 금지). 아래 필수 키를 반드시 포함:
   seed, accelerator, precision, devices, strategy, epochs, batch_size, optimizer, lr, weight_decay,
   gradient_clip, loss_function, data_dir, val_ratio, num_workers, primary_metric, checkpoint_dir
   추가 키는 flat으로 자유롭게 추가 가능. `model:` / `training:` / `data:` 같은 nested 그룹 금지.
   예시: `seed: 42`, `batch_size: 16`, `lr: 1.0e-4` (O) / `training: {{seed: 42}}` (X)
5. METRICS stdout 계약: print(f"METRICS:{{json.dumps({{...}})}}") — required_keys 모두 출력
6. **config 접근은 반드시 `config.get("key", default)` 사용** — `config["key"]` 직접 접근 절대 금지 (KeyError 방지)

아래 JSON으로만 출력 (코드 블록 없이):
{{
  "model_py": "...", "module_py": "...", "data_py": "...", "default_yaml": "...",
  "description": "...", "architecture_summary": "...", "param_estimate_M": 0.0
}}"""

    print(f"  [Step 1 / Claude] {task_family} 기반 코드 생성...")
    # claude-opus는 tool 호출을 시도하므로 sonnet 사용
    generated = None
    for _attempt in range(3):
        try:
            raw_response = query_claude(prompt, model="claude-sonnet-4-20250514")
            if not raw_response or len(raw_response) < 10:
                print(f"  [WARN] Response too short ({len(raw_response) if raw_response else 0}), retrying ({_attempt+1}/3)...")
                continue
            generated = parse_json(raw_response)
            break
        except Exception as e:
            print(f"  [WARN] Parse failed ({_attempt+1}/3): {e}")
            if _attempt == 2:
                raise
    if generated is None:
        raise RuntimeError("Base generation: 3회 시도 후 유효한 응답 없음")

    # generation metadata 강화
    code_analysis_strength = "high" if component_info else ("medium" if tips else "low")
    generated["generation_metadata"] = {
        "task_family": task_family,
        "generation_mode": "skeleton_grounded" if skeleton_path else (
            "reference_grounded" if code_analysis_strength == "high" else "pattern_grounded"),
        "reference_strength": code_analysis_strength,
        "used_pattern_ids": [p.get("pattern_id", "") for p in pattern_cands[:3]],
        "used_baselines": [b.get("name", "") for b in spec.get("synthesized_baselines", [])[:3]],
        "used_skeleton": skeleton_path,
        "must_not_do": must_not_do,
        "family_contract_version": "1.0",
    }
    return generated


# ──────────────────────────────────────────────────────────
# Path A 패치 context 정규화
# ──────────────────────────────────────────────────────────

def _prepare_accepted_patch_context(
    patches: list[dict],
    accepted_indexes: list[int],
) -> dict:
    """수락된 GPT 패치를 파일별로 그룹화한 구조화된 context를 생성한다.

    - 수락된 패치만 유지 (거절된 패치 완전 제거)
    - target_file별로 그룹화
    - rationale, hypothesis_alignment_check, changes 보존

    Returns:
        {
          "accepted_count": int,
          "by_file": {
            "model.py": [
              {"index": 0, "rationale": ..., "hypothesis_alignment_check": ..., "changes": [...]}
            ]
          },
          "target_files": ["model.py"]
        }
    """
    accepted_set = set(accepted_indexes)
    by_file: dict[str, list[dict]] = {}

    for i, patch in enumerate(patches):
        if i not in accepted_set:
            continue
        fname = patch.get("target_file", "unknown")
        if fname not in by_file:
            by_file[fname] = []
        by_file[fname].append({
            "index":                    i,
            "rationale":                patch.get("rationale", ""),
            "hypothesis_alignment_check": patch.get("hypothesis_alignment_check", ""),
            "changes":                  patch.get("changes", []),
        })

    return {
        "accepted_count": sum(len(v) for v in by_file.values()),
        "by_file":        by_file,
        "target_files":   list(by_file.keys()),
    }


# ──────────────────────────────────────────────────────────
# Step 1 (Path A) — Claude: 이전 패키지 기반 최소 revision
# ──────────────────────────────────────────────────────────

def _copy_previous_package(previous_pkg: Path, new_pkg: Path) -> None:
    """이전 패키지를 새 버전 디렉토리로 복사하고 runtime 아티팩트를 초기화한다.

    - 코드/설정/문서 파일: 이전 버전 그대로 보존 (revision 대상)
    - artifacts/ 하위 디렉토리: 내용 삭제 후 빈 디렉토리로 재생성
    - proposals/: 이전 제안에 "prev_" prefix 추가 (새 버전과 혼용 방지)
    """
    if new_pkg.exists():
        shutil.rmtree(new_pkg)
    new_pkg.mkdir(parents=True)

    # 필요한 파일/디렉토리만 명시적으로 복사 (v1, __pycache__, data/, artifacts/ 등 제외)
    _COPY_FILES = ["model.py", "module.py", "data.py", "train.py",
                   "experiment_spec.json"]
    _COPY_DIRS = ["configs", "scripts", "tests"]

    for f in _COPY_FILES:
        src = previous_pkg / f
        if src.exists():
            shutil.copy2(src, new_pkg / f)

    for d in _COPY_DIRS:
        src = previous_pkg / d
        if src.exists() and src.is_dir():
            shutil.copytree(src, new_pkg / d)

    # proposals/ 복사 + prev_ prefix (이전 GPT 패치/리뷰 이력 보존)
    prev_proposals = previous_pkg / "proposals"
    new_proposals = new_pkg / "proposals"
    new_proposals.mkdir(exist_ok=True)
    if prev_proposals.exists():
        for old_file in prev_proposals.glob("*.json"):
            new_name = f"prev_{old_file.name}" if not old_file.name.startswith("prev_") else old_file.name
            shutil.copy2(old_file, new_proposals / new_name)

    # ── artifacts/ 빈 디렉토리 구조 생성 ──────────────────
    for sub in ["checkpoints", "logs", "metrics"]:
        (new_pkg / "artifacts" / sub).mkdir(parents=True, exist_ok=True)

    print(f"  [Path A] 이전 패키지 복사: {previous_pkg.name} → {new_pkg.name}"
          f" (코드+설정만, artifacts 초기화)")


def _load_previous_package_context(previous_pkg: Path) -> dict:
    """이전 패키지의 편집 가능 파일과 메타데이터를 로드한다."""
    ctx: dict = {}
    for key, rel in [
        ("model_py",     "model.py"),
        ("module_py",    "module.py"),
        ("data_py",      "data.py"),
        ("default_yaml", "configs/default.yaml"),
    ]:
        fpath = previous_pkg / rel
        ctx[key] = fpath.read_text(encoding="utf-8") if fpath.exists() else ""

    spec_path = previous_pkg / "experiment_spec.json"
    ctx["experiment_spec"] = (
        json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    )

    result_path = previous_pkg / "result_summary.json"
    ctx["result_summary"] = (
        json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    )
    return ctx


def _claude_generate_revision_patch(
    previous_ctx: dict,
    hypothesis: dict,
    spec: dict,
    improvement_hints: str = "",
    patch_context: dict | None = None,
    gemini_review: dict | None = None,
) -> dict:
    """이전 패키지를 기반으로 최소한의 수정만 적용한 revision을 생성한다.

    Args:
        patch_context: _prepare_accepted_patch_context()가 반환한 구조화된 패치 context.
                       by_file별로 그룹화되어 있으며, 수락된 패치만 포함.
    Returns:
        수정된 파일 + applied_patch_indexes, skipped_patch_indexes, patch_application_notes
    """
    hyp          = hypothesis.get("hypothesis", {})
    primary      = spec["evaluation_config"]["primary_metric"]
    prev_result  = previous_ctx.get("result_summary", {})
    prev_primary = prev_result.get("primary_metric", {})

    prev_result_section = ""
    if prev_primary:
        prev_result_section = (
            f"\n## 이전 실험 결과\n"
            f"- {prev_primary.get('name', primary)}: {prev_primary.get('value', 'N/A')} "
            f"(target={prev_primary.get('target', 'N/A')}, met={prev_primary.get('met', False)})\n"
            f"- status: {prev_result.get('status', 'unknown')}"
        )

    # GPT 패치 context: 파일별 구조화 형태로 표현
    gpt_section = ""
    if patch_context and patch_context.get("accepted_count", 0) > 0:
        by_file     = patch_context.get("by_file", {})
        target_files = patch_context.get("target_files", [])
        patch_lines  = [
            f"\n### {fname} ({len(patches)}개 패치)\n"
            + "\n".join(
                f"  [patch {p['index']}] {p['rationale']}\n"
                f"    alignment: {p['hypothesis_alignment_check']}\n"
                f"    changes: {json.dumps(p['changes'], ensure_ascii=False)}"
                for p in patches
            )
            for fname, patches in by_file.items()
        ]
        gpt_section = (
            f"\n## 적용 예정 GPT 패치 (수락된 것만, 파일별 그룹화)\n"
            f"대상 파일: {target_files}\n"
            + "\n".join(patch_lines)
        )

    gemini_section = ""
    if gemini_review:
        gemini_section = (
            f"\n## Gemini 분석\n"
            f"- root_cause: {gemini_review.get('root_cause_analysis', '')}\n"
            f"- suggestions: {gemini_review.get('improvement_suggestions', [])}"
        )

    arch    = spec["model_architecture"]
    tr_cfg  = spec["training_config"]
    ev_cfg  = spec["evaluation_config"]
    out_ctr = spec.get("output_contract", {})

    prompt = f"""당신은 실험 패키지 revision 담당자이자 **유일한 코드 작성자(Claude 역할)**입니다.
이전 패키지 코드를 기반으로 **최소한의 변경(minimal diff)**만 적용하여 개선하세요.
불필요한 리팩토링, 파일 이동, 전체 재작성은 금지입니다.
{prev_result_section}

## [최우선] 구현 계약 (experiment_spec — 이 버전의 활성 계약)
- spec_id: {spec['spec_id']}
- model_architecture: name={arch['name']}, key_components={arch.get('key_components', [])}, param_budget_M≤{arch.get('param_budget_M','N/A')}M
- training_config: optimizer={tr_cfg['optimizer']}, lr={tr_cfg['lr']}, loss={tr_cfg['loss_function']}
- evaluation_config: {primary} ≥ {ev_cfg['target_value']}
- output_contract: required_keys={out_ctr.get('required_keys', [primary])}

## [과학적 기준] 가설 (변경 금지 — 방향 참고용)
{hyp.get('statement_kr', hyp.get('statement', ''))}

## 개선 목표
- 목표: {primary} ≥ {ev_cfg['target_value']}
- 개선 요청: {improvement_hints}
{gpt_section}{gemini_section}

## 현재 model.py
```python
{previous_ctx.get('model_py', '')[:3000]}
```

## 현재 module.py
```python
{previous_ctx.get('module_py', '')[:2000]}
```

## 현재 data.py
```python
{previous_ctx.get('data_py', '')[:2000]}
```

## 현재 default.yaml
```yaml
{previous_ctx.get('default_yaml', '')[:1000]}
```

## 수정 규칙
1. 수정이 필요한 파일만 변경 — 변경 없는 파일은 빈 문자열로 반환
2. train.py 수정 금지
3. METRICS:{{...}} stdout 계약 보존 (metric 키 이름 변경 금지)
4. GPT 패치: 현재 코드와 호환되는 것만 적용, 호환 안 되면 건너뜀
5. 파일 전체 재작성 금지 — 변경 필요한 부분만 수정
6. 인덱스 규칙 (중요):
   - applied_patch_indexes / skipped_patch_indexes는 반드시 원본 GPT proposal 인덱스를 사용한다
   - 로컬 순서로 0, 1, 2... 재번호 매기기 금지
   - 예: GPT가 proposal 인덱스 1, 4, 7을 제안했고 1과 4를 적용, 7을 건너뜀 →
         applied_patch_indexes: [1, 4]  (원본 인덱스)
         skipped_patch_indexes: [7]     (원본 인덱스)
   - patch_application_notes에도 원본 인덱스 번호로 기재할 것

아래 JSON으로만 출력:
{{
  "model_py": "수정된 전체 내용 (변경 없으면 빈 문자열)",
  "module_py": "수정된 전체 내용 (변경 없으면 빈 문자열)",
  "data_py": "수정된 전체 내용 (변경 없으면 빈 문자열)",
  "default_yaml": "수정된 전체 내용 (변경 없으면 빈 문자열)",
  "change_summary": "변경 내용 요약",
  "files_changed": ["model.py"],
  "files_unchanged": ["module.py", "data.py"],
  "applied_patch_indexes": [1, 4],
  "skipped_patch_indexes": [7],
  "patch_application_notes": "패치 1(원본idx): 적용됨. 패치 4(원본idx): 적용됨. 패치 7(원본idx): 건너뜀 (API 불일치)",
  "description": "...", "architecture_summary": "...", "param_estimate_M": 0.0
}}"""

    print("  [Step 1-A / Claude] 최소 revision 패치 생성...")
    result = None
    for _attempt in range(3):
        try:
            raw = query_claude(prompt, model="claude-sonnet-4-20250514")
            if not raw or not raw.strip():
                print(f"    [Revision] 빈 응답 (시도 {_attempt+1}/3), 재시도...")
                continue
            result = parse_json(raw)
            break
        except Exception as e:
            print(f"    [Revision] 파싱 실패 (시도 {_attempt+1}/3): {e}")
            if _attempt == 2:
                raise
    if result is None:
        raise RuntimeError("Revision patch: 3회 시도 후 빈 응답")

    # 빈 문자열 = 변경 없음 → 이전 파일 내용 그대로 유지
    for key in ("model_py", "module_py", "data_py", "default_yaml"):
        if not result.get(key):
            result[key] = previous_ctx.get(key, "")

    changed   = result.get("files_changed", [])
    unchanged = result.get("files_unchanged", [])
    applied   = result.get("applied_patch_indexes", [])
    skipped   = result.get("skipped_patch_indexes", [])
    print(f"    변경: {changed}  유지: {unchanged}")
    print(f"    패치 적용: {applied}  건너뜀: {skipped}")
    if result.get("patch_application_notes"):
        print(f"    노트: {result['patch_application_notes']}")
    return result


# ──────────────────────────────────────────────────────────
# Step 2 — GPT/Codex: 패치 제안
# ──────────────────────────────────────────────────────────

def _gpt_propose_patches(
    generated: dict,
    spec: dict,
    hypothesis: dict,
) -> dict:
    """GPT/Codex가 model.py / module.py / data.py에 대한 패치를 제안한다."""
    hyp        = hypothesis.get("hypothesis", {})
    primary    = spec["evaluation_config"]["primary_metric"]
    param_budget = spec["model_architecture"]["param_budget_M"]

    arch    = spec["model_architecture"]
    ev_cfg  = spec["evaluation_config"]
    out_ctr = spec.get("output_contract", {})

    system_msg = (
        "You are a PyTorch code patch reviewer (GPT/Codex role). "
        "Your ONLY job is to propose TARGETED patches to improve the generated code. "
        "RULES: "
        "(1) Propose only minimal targeted patches — full-file rewrites are FORBIDDEN. "
        "(2) Each patch must be linked to a specific experiment_spec field. "
        "(3) Do NOT rewrite train.py or the module.py training loop. "
        "(4) Claude is the sole code author — you only suggest; Claude decides. "
        "Return valid JSON only."
    )

    user_msg = f"""Review the PyTorch experiment code below and propose spec-driven targeted patches.

## Experiment Spec (implementation contract — patches must align with this)
- spec_id: {spec['spec_id']}
- model_architecture.name: {arch['name']}
- model_architecture.key_components: {arch.get('key_components', [])}
- model_architecture.param_budget_M: ≤ {param_budget}M
- evaluation_config.primary_metric: {primary} ≥ {ev_cfg['target_value']}
- evaluation_config.secondary_metrics: {ev_cfg.get('secondary_metrics', [])}
- output_contract.required_keys: {out_ctr.get('required_keys', [primary])}

## Hypothesis (scientific truth — do not contradict)
{hyp.get('statement_kr', hyp.get('statement', ''))}

## Hypothesis Contract (patches must align with these)
- mechanism: {spec.get('hypothesis_contract', {}).get('mechanism', '')}
- target_metric_raw: {spec.get('hypothesis_contract', {}).get('target_metric_raw', '')}
- constraints_raw: {spec.get('hypothesis_contract', {}).get('constraints_raw', '')}

## Generated model.py
```python
{generated.get('model_py', '')[:3000]}
```

## Generated module.py
```python
{generated.get('module_py', '')[:2000]}
```

## Generated data.py
```python
{generated.get('data_py', '')[:2000]}
```

Propose ONLY targeted patches. Full-file rewrites are forbidden.
Each patch MUST include spec_field, hypothesis_alignment_check, and breaks_comparability.
Return JSON only:
{{
  "patches": [
    {{
      "target_file": "model.py",
      "spec_field": "model_architecture.key_components",
      "rationale": "...",
      "hypothesis_alignment_check": "...",
      "breaks_comparability": false,
      "complexity_delta_loc": 5,
      "changes": [
        {{"type": "replace", "old": "exact old code snippet", "new": "new code snippet"}}
      ]
    }}
  ],
  "overall_assessment": "...",
  "critical_issues": []
}}"""

    print("  [Step 2 / GPT]   패치 제안...")
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
        print(f"    → {len(result.get('patches', []))}개 패치 제안")
        return result
    except Exception as e:
        print(f"    [경고] GPT 패치 제안 실패: {e} → 빈 제안으로 계속")
        return {"patches": [], "overall_assessment": f"GPT call failed: {e}", "critical_issues": []}


# ──────────────────────────────────────────────────────────
# Step 3 — Gemini: 설계 리뷰
# ──────────────────────────────────────────────────────────

def _gemini_design_review(
    generated: dict,
    spec: dict,
    hypothesis: dict,
) -> dict:
    """Gemini가 아키텍처 설계와 가설 정합성을 리뷰한다."""
    hyp    = hypothesis.get("hypothesis", {})
    primary = spec["evaluation_config"]["primary_metric"]

    arch   = spec["model_architecture"]
    ev_cfg = spec["evaluation_config"]

    prompt = f"""You are a deep learning design reviewer (Gemini role).
Your ONLY job is to review architecture design and hypothesis alignment.
CRITICAL RULES:
  - Do NOT write code. Do NOT suggest direct code edits.
  - Do NOT act as a second patch engine or code author.
  - Only provide structured design critique that Claude (the code author) can interpret.
Return valid JSON only.

## Experiment Spec (implementation contract)
- spec_id: {spec['spec_id']}
- model_architecture.name: {arch['name']}
- model_architecture.key_components: {arch.get('key_components', [])}
- model_architecture.param_budget_M: ≤ {arch.get('param_budget_M', 'N/A')}M
- evaluation_config.primary_metric: {primary} ≥ {ev_cfg['target_value']}

## Hypothesis (scientific truth)
{hyp.get('statement_kr', hyp.get('statement', ''))}
Key mechanism: {hyp.get('expected_mechanism', hyp.get('key_mechanism', hyp.get('mechanism', '')))}

## Architecture Summary
{generated.get('architecture_summary', '')}

## model.py (for design review only — do NOT propose code changes)
```python
{generated.get('model_py', '')[:2500]}
```

## Hypothesis Contract (verify implementation)
- mechanism: {spec.get('hypothesis_contract', {}).get('mechanism', '')}
- target_metric_raw: {spec.get('hypothesis_contract', {}).get('target_metric_raw', '')}
- constraints_raw: {spec.get('hypothesis_contract', {}).get('constraints_raw', '')}

Review ONLY the following dimensions:
1. mechanism_check: does the architecture actually implement the hypothesis mechanism? Is the mechanism concretized in code-level design elements (not just described in comments)?
2. spec_alignment: does the design match spec.model_architecture fields?
3. metric_evaluation_path: does the code contain a clear evaluation path to achieve the target_metric? Are the correct METRICS keys present?
4. constraints_compliance: are constraints reflected in architecture size, training config, and default.yaml?
5. missing_components: what critical components are absent?
6. unnecessary_complexity: what adds complexity without serving the hypothesis?
7. bottlenecks: where is the most likely experimental failure point?

Return JSON only:
{{
  "verdict": "accept_as_is | accept_with_patch | revise_experiment",
  "hypothesis_alignment_score": 0.0,
  "mechanism_check": "does the architecture implement the key mechanism? explain",
  "spec_alignment": "does the design match the spec fields? explain",
  "metric_evaluation_path": "does the code contain evaluation path for target_metric? explain",
  "constraints_compliance": "are constraints reflected in architecture/config? explain",
  "issues": [
    {{"severity": "critical|major|minor", "description": "...", "design_suggestion": "..."}}
  ],
  "missing_components": [],
  "unnecessary_complexity": [],
  "bottlenecks": [],
  "overall_comment": "..."
}}"""

    print("  [Step 3 / Gemini] 설계 리뷰...")
    try:
        model = get_gemini_model()
        resp  = model.generate_content(prompt)
        text  = resp.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        result = json.loads(text)
        print(f"    → verdict={result.get('verdict')}, "
              f"alignment={result.get('hypothesis_alignment_score')}")
        return result
    except Exception as e:
        print(f"    [경고] Gemini 리뷰 실패: {e} → 기본 리뷰로 계속")
        return {
            "verdict": "accept_as_is",
            "hypothesis_alignment_score": 0.7,
            "issues": [],
            "missing_components": [],
            "unnecessary_complexity": [],
            "overall_comment": f"Gemini call failed: {e}",
        }


# ──────────────────────────────────────────────────────────
# Step 3-B — Gemini: Patch Ballot (패치별 수락/거절 투표)
# ──────────────────────────────────────────────────────────

def _gemini_patch_ballot(
    patches: list[dict],
    spec: dict,
    hypothesis: dict,
    code_files: dict,
) -> list[dict]:
    """Gemini가 GPT 패치 각각에 대해 수락/거절 투표를 수행한다.

    기존 설계 리뷰와 별도로, 패치 단위 판정만 수행.
    코드 작성/패치 재작성은 하지 않으며 수락/거절 판단만 반환한다.

    Returns:
        패치별 투표 결과 리스트 (patch_index, vote, reason, ...)
    """
    if not patches:
        return []

    hyp     = hypothesis.get("hypothesis", {})
    arch    = spec["model_architecture"]
    ev_cfg  = spec["evaluation_config"]
    primary = ev_cfg["primary_metric"]

    # 코드 컨텍스트 (토큰 절약)
    code_context = "\n\n".join(
        f"## {name}\n```\n{content[:2000]}\n```"
        for name, content in code_files.items()
        if content
    )

    patches_json = json.dumps(
        [{"index": i, **p} for i, p in enumerate(patches)],
        ensure_ascii=False, indent=2,
    )

    prompt = f"""You are a deep learning design reviewer (Gemini role).
Your job is to review EACH GPT patch proposal independently and vote accept or reject.
Do NOT write code. Do NOT propose alternative patches. Only vote on each patch.

## Experiment Spec
- model_architecture.name: {arch['name']}
- model_architecture.key_components: {arch.get('key_components', [])}
- model_architecture.param_budget_M: <= {arch.get('param_budget_M', 'N/A')}M
- evaluation_config.primary_metric: {primary} >= {ev_cfg['target_value']}

## Hypothesis
{hyp.get('statement_kr', hyp.get('statement', ''))}
Key mechanism: {hyp.get('expected_mechanism', hyp.get('key_mechanism', hyp.get('mechanism', '')))}

## Hypothesis Contract
- mechanism: {spec.get('hypothesis_contract', {}).get('mechanism', '')}
- constraints_raw: {spec.get('hypothesis_contract', {}).get('constraints_raw', '')}

## Current Code
{code_context}

## GPT Patches to Review
{patches_json}

For EACH patch, evaluate:
1. spec_compatibility: does it comply with the experiment spec?
2. hypothesis_alignment: does it align with the hypothesis mechanism?
3. comparability_risk: does it break result comparability? (low/medium/high)
4. implementation_feasibility: can it be cleanly integrated?
5. expected_gain_vs_complexity: is the benefit worth the complexity?

Return JSON only — an array of per-patch votes:
[
  {{
    "patch_index": 0,
    "vote": "accept|reject|weak_accept|weak_reject",
    "reason": "concise reason for this vote",
    "spec_risk": "what spec field is at risk, if any",
    "mechanism_alignment": "how this patch relates to the hypothesis mechanism",
    "comparability_risk": "low|medium|high"
  }}
]"""

    print("  [Step 3-B / Gemini] 패치별 투표...")
    try:
        model = get_gemini_model()
        resp  = model.generate_content(prompt)
        text  = resp.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        result = json.loads(text)
        # 리스트가 아닌 경우 래핑
        if isinstance(result, dict):
            result = result.get("votes", result.get("ballots", [result]))
        votes_summary = ", ".join(
            f"patch {v.get('patch_index','?')}={v.get('vote','?')}" for v in result
        )
        print(f"    -> {len(result)}개 투표: {votes_summary}")
        return result
    except Exception as e:
        print(f"    [경고] Gemini 패치 투표 실패: {e} -> 전체 weak_accept 기본값")
        return [
            {
                "patch_index": i,
                "vote": "weak_accept",
                "reason": f"Gemini ballot failed: {e}",
                "spec_risk": "",
                "mechanism_alignment": "unknown",
                "comparability_risk": "low",
            }
            for i in range(len(patches))
        ]


# ──────────────────────────────────────────────────────────
# Step 3-C — Claude: Patch Ballot (패치별 독립 수락/거절 판정)
# ──────────────────────────────────────────────────────────

def _claude_patch_ballot(
    patches: list[dict],
    spec: dict,
    hypothesis: dict,
    code_files: dict,
) -> list[dict]:
    """Claude가 merge 전에 GPT 패치 각각에 대해 독립적인 수락/거절 판정을 수행한다.

    이 단계에서는 코드 전체를 다시 쓰지 않고, 패치별 판정만 수행한다.
    merge 단계와 분리되어 ballot 결과가 2-of-3 규칙에 사용된다.

    Returns:
        패치별 투표 결과 리스트 (patch_index, vote, reason, ...)
    """
    if not patches:
        return []

    hyp     = hypothesis.get("hypothesis", {})
    arch    = spec["model_architecture"]
    ev_cfg  = spec["evaluation_config"]
    primary = ev_cfg["primary_metric"]

    # 코드 컨텍스트
    code_context = "\n\n".join(
        f"## {name}\n```\n{content[:2000]}\n```"
        for name, content in code_files.items()
        if content
    )

    patches_json = json.dumps(
        [{"index": i, **p} for i, p in enumerate(patches)],
        ensure_ascii=False, indent=2,
    )

    prompt = f"""당신은 실험 패키지의 코드 작성자(Claude 역할)입니다.
이 단계에서는 코드를 작성하지 말고, GPT가 제안한 각 패치에 대해 독립적인 수락/거절 판정만 수행하세요.
판정 기준: spec 호환성, 가설 정합성, 결과 비교 가능성, 구현 가능성, 이득 대비 복잡도.

## Experiment Spec
- model_architecture.name: {arch['name']}
- model_architecture.key_components: {arch.get('key_components', [])}
- model_architecture.param_budget_M: <= {arch.get('param_budget_M', 'N/A')}M
- evaluation_config.primary_metric: {primary} >= {ev_cfg['target_value']}

## 가설
{hyp.get('statement_kr', hyp.get('statement', ''))}
핵심 메커니즘: {hyp.get('expected_mechanism', hyp.get('key_mechanism', hyp.get('mechanism', '')))}

## Hypothesis Contract
- mechanism: {spec.get('hypothesis_contract', {}).get('mechanism', '')}
- constraints_raw: {spec.get('hypothesis_contract', {}).get('constraints_raw', '')}

## 현재 코드
{code_context}

## GPT 패치 제안 (판정 대상)
{patches_json}

각 패치에 대해 판정하세요. 코드 작성 금지 — 판정만 수행.

아래 JSON 배열로만 출력:
[
  {{
    "patch_index": 0,
    "vote": "accept|reject|weak_accept|weak_reject",
    "reason": "판정 이유",
    "spec_risk": "위험한 spec 필드 (있는 경우)",
    "mechanism_alignment": "가설 메커니즘과의 관계",
    "comparability_risk": "low|medium|high"
  }}
]"""

    print("  [Step 3-C / Claude] 패치별 독립 판정...")
    try:
        raw = query_claude(prompt, model="claude-sonnet-4-20250514")
        result = parse_json(raw)
        # dict로 래핑된 경우 처리
        if isinstance(result, dict):
            result = result.get("votes", result.get("ballots", [result]))
        votes_summary = ", ".join(
            f"patch {v.get('patch_index','?')}={v.get('vote','?')}" for v in result
        )
        print(f"    -> {len(result)}개 판정: {votes_summary}")
        return result
    except Exception as e:
        print(f"    [경고] Claude 패치 판정 실패: {e} -> 전체 weak_accept 기본값")
        return [
            {
                "patch_index": i,
                "vote": "weak_accept",
                "reason": f"Claude ballot failed: {e}",
                "spec_risk": "",
                "mechanism_alignment": "unknown",
                "comparability_risk": "low",
            }
            for i in range(len(patches))
        ]


# ──────────────────────────────────────────────────────────
# Step 3-D — 2-of-3 Patch Merge Decision (패치별 합의 규칙)
# ──────────────────────────────────────────────────────────

# 투표 분류 상수
_ACCEPT_VOTES = frozenset({"accept", "weak_accept"})
_REJECT_VOTES = frozenset({"reject", "weak_reject"})


def _compute_patch_decisions(
    gpt_patches: list[dict],
    gemini_ballot: list[dict],
    claude_ballot: list[dict],
) -> list[dict]:
    """2-of-3 합의 규칙으로 패치별 최종 채택/기각을 결정한다.

    투표 입력:
      - GPT: 패치를 제안했으므로 implicit accept
      - Gemini: patch ballot 결과
      - Claude: patch ballot 결과

    합의 규칙:
      1. accept/weak_accept가 2개 이상 -> accept
      2. reject/weak_reject가 2개 이상 -> reject
      3. comparability_risk=high인 투표가 하나라도 있으면 -> reject (hard rule)
      4. spec_field가 critical이고 Gemini+Claude 모두 reject -> reject (hard rule)
      5. 그 외 -> ambiguous (Claude tie-break 허용)

    Returns:
        패치별 merge_decision 리스트
    """
    # 인덱스별 투표 매핑
    gemini_by_idx = {v.get("patch_index", i): v for i, v in enumerate(gemini_ballot)}
    claude_by_idx = {v.get("patch_index", i): v for i, v in enumerate(claude_ballot)}

    decisions: list[dict] = []

    for i, patch in enumerate(gpt_patches):
        gpt_vote = "accept"  # GPT는 제안자이므로 implicit accept
        gem_vote_data = gemini_by_idx.get(i, {})
        cla_vote_data = claude_by_idx.get(i, {})
        gem_vote = gem_vote_data.get("vote", "weak_accept")
        cla_vote = cla_vote_data.get("vote", "weak_accept")

        # 투표 집계
        votes = [gpt_vote, gem_vote, cla_vote]
        accept_count = sum(1 for v in votes if v in _ACCEPT_VOTES)
        reject_count = sum(1 for v in votes if v in _REJECT_VOTES)

        # comparability_risk=high hard rule 검사
        comp_risks = [
            gem_vote_data.get("comparability_risk", "low"),
            cla_vote_data.get("comparability_risk", "low"),
        ]
        has_high_comp_risk = any(r == "high" for r in comp_risks)

        # spec_field critical + Gemini&Claude 모두 reject -> hard reject
        spec_field = patch.get("spec_field", "")
        is_critical_spec = "critical" in spec_field.lower() if spec_field else False
        both_reject = gem_vote in _REJECT_VOTES and cla_vote in _REJECT_VOTES

        # 최종 결정
        if has_high_comp_risk:
            final_decision = "reject"
            decision_rule = "comparability_risk=high (hard rule)"
        elif is_critical_spec and both_reject:
            final_decision = "reject"
            decision_rule = "critical spec_field + Gemini&Claude both reject (hard rule)"
        elif accept_count >= 2:
            final_decision = "accept"
            decision_rule = f"2-of-3 accept ({accept_count} accept votes)"
        elif reject_count >= 2:
            final_decision = "reject"
            decision_rule = f"2-of-3 reject ({reject_count} reject votes)"
        else:
            final_decision = "ambiguous"
            decision_rule = "no clear majority — Claude tie-break allowed"

        decisions.append({
            "patch_index": i,
            "target_file": patch.get("target_file", "unknown"),
            "spec_field": spec_field,
            "gpt_proposed": True,
            "gemini_vote": gem_vote,
            "claude_vote": cla_vote,
            "gemini_reason": gem_vote_data.get("reason", ""),
            "claude_reason": cla_vote_data.get("reason", ""),
            "comparability_risk": max(comp_risks, key=lambda r: {"low": 0, "medium": 1, "high": 2}.get(r, 0)),
            "final_decision": final_decision,
            "decision_rule": decision_rule,
        })

    accepted = sum(1 for d in decisions if d["final_decision"] == "accept")
    rejected = sum(1 for d in decisions if d["final_decision"] == "reject")
    ambiguous = sum(1 for d in decisions if d["final_decision"] == "ambiguous")
    print(f"  [2-of-3 Merge] {len(decisions)}개 패치: "
          f"accept={accepted}, reject={rejected}, ambiguous={ambiguous}")

    return decisions


# ──────────────────────────────────────────────────────────
# Step 4 — Claude: Merge (합의 결과 집행자)
# ──────────────────────────────────────────────────────────

def _claude_merge(
    generated: dict,
    gpt_proposal: dict,
    gemini_review: dict,
    spec: dict,
    hypothesis: dict,
    merge_decisions: list[dict] | None = None,
) -> dict:
    """Claude가 2-of-3 합의 결과를 집행하여 최종 파일을 확정한다.

    merge_decisions가 제공되면:
      - final_decision == "accept" 패치만 적용
      - final_decision == "ambiguous" 패치는 Claude tie-break 후 적용 여부 결정
      - final_decision == "reject" 패치는 제외
    merge_decisions가 없으면 기존 로직(하위 호환) 사용.
    """
    hyp          = hypothesis.get("hypothesis", {})
    primary      = spec["evaluation_config"]["primary_metric"]
    param_budget = spec["model_architecture"]["param_budget_M"]

    gpt_patches = gpt_proposal.get("patches", [])
    gemini_issues = [
        i for i in gemini_review.get("issues", [])
        if i.get("severity") in ("critical", "major")
    ]

    arch    = spec["model_architecture"]
    ev_cfg  = spec["evaluation_config"]
    out_ctr = spec.get("output_contract", {})

    # ── 2-of-3 ballot 결과에 따른 패치 필터링 ──────────────────
    # merge_decisions가 있으면 합의 결과를 집행; 없으면 전체 패치를 Claude에게 전달 (하위 호환)
    accepted_patches: list[dict] = []
    ambiguous_patches: list[dict] = []
    rejected_patches: list[dict] = []

    if merge_decisions:
        decision_by_idx = {d["patch_index"]: d for d in merge_decisions}
        for i, patch in enumerate(gpt_patches):
            dec = decision_by_idx.get(i, {})
            fd = dec.get("final_decision", "accept")
            enriched = {**patch, "ballot_decision": dec}
            if fd == "accept":
                accepted_patches.append(enriched)
            elif fd == "ambiguous":
                ambiguous_patches.append(enriched)
            else:
                rejected_patches.append(enriched)
    else:
        # 하위 호환: ballot 없으면 전체 패치를 수락 후보로 처리
        accepted_patches = list(gpt_patches)

    # Claude에게 보여줄 패치 (accept + ambiguous만)
    mergeable_patches = accepted_patches + ambiguous_patches

    # 합의 결과 섹션 (merge_decisions가 있는 경우)
    ballot_section = ""
    if merge_decisions:
        ballot_summary = json.dumps(merge_decisions, ensure_ascii=False, indent=2)
        ballot_section = f"""
## [2-of-3 Patch Ballot 결과] — 합의 결과를 집행하세요
아래 합의 결과를 반드시 존중하세요:
- final_decision == "accept": 반드시 적용
- final_decision == "ambiguous": 당신이 tie-break로 적용 여부를 결정 가능
- final_decision == "reject": 적용 금지 (이 패치는 아래 목록에서 제외됨)

rejected 패치 ({len(rejected_patches)}개): {[r.get('ballot_decision', {}).get('patch_index') for r in rejected_patches]}

합의 상세:
{ballot_summary}
"""

    prompt = f"""당신은 실험 패키지의 최종 통합자이자 **유일한 코드 작성자(Claude 역할)**입니다.
당신은 합의 결과(2-of-3 patch ballot)의 **집행자**입니다.
합의에서 reject된 패치를 몰래 적용하거나, 합의 외의 독단적 코드 변경을 하면 안 됩니다.
새로운 아이디어가 있다면 merge_log에 pseudo_patch로 별도 기록하세요.
{ballot_section}
## [최우선] 구현 계약 (experiment_spec — 모든 결정의 기준)
- spec_id: {spec['spec_id']}
- model_architecture: name={arch['name']}, key_components={arch.get('key_components', [])}, param_budget_M<={param_budget}M
- evaluation_config: {primary} >= {ev_cfg['target_value']}, secondary={ev_cfg.get('secondary_metrics', [])}
- output_contract: required_keys={out_ctr.get('required_keys', [primary])}, stdout_pattern={out_ctr.get('stdout_pattern', '')}

## [과학적 기준] 연구 가설
{hyp.get('statement_kr', hyp.get('statement', ''))}

## 현재 코드 (Claude 초안)
### model.py
```python
{generated.get('model_py', '')}
```
### module.py
```python
{generated.get('module_py', '')}
```
### data.py
```python
{generated.get('data_py', '')}
```
### default.yaml
```yaml
{generated.get('default_yaml', '')}
```

## 적용 대상 GPT/Codex 패치 ({len(mergeable_patches)}개) — reject 패치 제외됨
{json.dumps(mergeable_patches, ensure_ascii=False, indent=2)}

## Gemini 설계 리뷰 — 설계 비평만, 코드 제안 아님
- verdict: {gemini_review.get('verdict')}
- mechanism_check: {gemini_review.get('mechanism_check', '')}
- spec_alignment: {gemini_review.get('spec_alignment', '')}
- alignment_score: {gemini_review.get('hypothesis_alignment_score')}
- critical/major issues: {json.dumps(gemini_issues, ensure_ascii=False)}
- missing_components: {gemini_review.get('missing_components', [])}
- bottlenecks: {gemini_review.get('bottlenecks', [])}

## Merge 규칙 (합의 집행)
1. [합의 존중] accept 패치는 반드시 적용. reject 패치는 적용 금지.
2. [ambiguous tie-break] ambiguous 패치만 당신이 적용 여부를 결정 가능.
3. [spec 호환성] spec.model_architecture / training_config / output_contract 위반 금지.
4. [가설 정합성] 가설 메커니즘을 구현하거나 보강하는 방향으로.
5. [output contract 보존] METRICS stdout 계약 및 required_keys 유지.
6. [pseudo_patch] 새 아이디어는 merge_log에 source="claude_pseudo_patch"로 기록.

## 지시사항
- train.py 절대 수정 금지
- METRICS stdout 계약 보존 (required_keys 이름 변경 금지)
- 패치별 merge 결과를 merge_log에 spec_field, ballot 투표 상세와 함께 기록

아래 JSON으로만 출력:
{{
  "model_py": "최종 model.py 전체",
  "module_py": "최종 module.py 전체",
  "data_py": "최종 data.py 전체",
  "default_yaml": "최종 default.yaml 전체",
  "merge_log": [
    {{
      "patch_index": 0,
      "source": "GPT|Gemini|claude_pseudo_patch",
      "item": "...",
      "spec_field": "...",
      "gpt_proposed": true,
      "gemini_vote": "accept|reject|weak_accept|weak_reject",
      "claude_vote": "accept|reject|weak_accept|weak_reject",
      "final_decision": "accept|reject|ambiguous",
      "decision_rule": "...",
      "merge_applied": true,
      "accepted": true,
      "reason": "..."
    }}
  ],
  "description": "최종 모델 설명",
  "architecture_summary": "한 줄 요약",
  "param_estimate_M": 0.0
}}"""

    print("  [Step 4 / Claude] Merge & 최종 확정 (합의 집행)...")
    for _attempt in range(3):
        try:
            raw = query_claude(prompt, model="claude-sonnet-4-20250514")
            if not raw or not raw.strip():
                print(f"    [Merge] 빈 응답 (시도 {_attempt+1}/3), 재시도...")
                continue
            return parse_json(raw)
        except Exception as e:
            print(f"    [Merge] 파싱 실패 (시도 {_attempt+1}/3): {e}")
            if _attempt == 2:
                raise


# ──────────────────────────────────────────────────────────
# Hypothesis Implementation Audit (mechanism / metric / constraints)
# ──────────────────────────────────────────────────────────

def _load_code_files(pkg_dir: Path) -> dict[str, str]:
    """패키지의 편집 가능 코드 파일과 config를 로드한다."""
    code_files: dict[str, str] = {}
    for key, rel in [
        ("model_py",     "model.py"),
        ("module_py",    "module.py"),
        ("data_py",      "data.py"),
        ("default_yaml", "configs/default.yaml"),
        ("train_py",     "train.py"),
    ]:
        fpath = pkg_dir / rel
        code_files[key] = fpath.read_text(encoding="utf-8") if fpath.exists() else ""
    return code_files


def _mechanism_audit(mechanism: str, code_files: dict, architecture_summary: str) -> dict:
    """Claude 기반 감사: mechanism이 실제 코드에 구현되었는가?

    mechanism 문자열이 코드의 구체적 설계 요소(아키텍처/학습/데이터 처리)에
    대응되는지 Claude에게 질의하여 판단한다.

    Returns:
        {
            "implemented": bool,
            "evidence": [...],
            "missing_links": [...],
            "risk_level": "low|medium|high",
            "mechanism_mapping": {...}
        }
    """
    if not mechanism or not mechanism.strip():
        return {
            "implemented": True,
            "evidence": ["mechanism 필드가 비어 있어 감사 생략"],
            "missing_links": [],
            "risk_level": "low",
            "mechanism_mapping": {},
        }

    # 코드 컨텍스트 구성 (토큰 절약을 위해 길이 제한)
    code_context = "\n\n".join(
        f"## {name}\n```\n{content[:2500]}\n```"
        for name, content in code_files.items()
        if content
    )

    prompt = f"""당신은 딥러닝 연구 감사관입니다. 아래 mechanism이 코드에 **실제로 작동하도록** 구현되었는지 판단하세요.

## Mechanism (가설에서 주장하는 핵심 작동 원리)
{mechanism}

## Architecture Summary
{architecture_summary}

## 코드 파일
{code_context}

## 감사 질문 (5단계 심층 검증)
1. **존재 여부**: mechanism이 코드에 클래스/함수/모듈로 정의되어 있는가?
2. **forward pass 참여**: mechanism 코드가 모델의 forward() 호출 경로에 포함되는가?
   - 정의만 되고 호출되지 않는 dead code는 "미구현"으로 판정
   - 주석이나 docstring에만 있는 것은 "미구현"
3. **학습 영향**: mechanism이 gradient를 받는 학습 가능한 파라미터를 포함하는가?
   - 또는 loss/data pipeline에 영향을 주는 요소인가?
4. **config 연동**: mechanism의 핵심 하이퍼파라미터가 default.yaml에서 설정 가능한가?
5. **baseline 구별**: mechanism을 제거하면 코드가 일반적 baseline과 동일해지는가?
   - 즉, mechanism이 이 가설을 고유하게 만드는 차별화 요소인가?

## 판정 기준
- implemented=true: 질문 1-3 모두 yes + 질문 5에서 baseline과 구별됨
- implemented=false: 질문 1-3 중 하나라도 no
- risk_level: low (1-5 모두 yes), medium (4-5 중 일부 no), high (1-3 중 일부 no)

아래 JSON으로만 출력:
{{
  "implemented": true,
  "evidence": ["코드에서 mechanism 구현을 확인한 구체적 증거 (클래스명, forward() 호출 경로 등)"],
  "missing_links": ["mechanism 중 코드에서 찾을 수 없거나 dead code인 요소"],
  "risk_level": "low|medium|high",
  "mechanism_mapping": {{
    "mechanism_concept": "대응하는 코드 요소 (클래스명, 함수명, 설정 키 등)"
  }},
  "forward_path_verified": true,
  "differentiates_from_baseline": true
}}"""

    try:
        result = parse_json(query_claude(prompt, model="claude-sonnet-4-20250514"))
        # 필수 필드 보장
        result.setdefault("implemented", False)
        result.setdefault("evidence", [])
        result.setdefault("missing_links", [])
        result.setdefault("risk_level", "high")
        result.setdefault("mechanism_mapping", {})
    except Exception as e:
        result = {
            "implemented": False,
            "evidence": [],
            "missing_links": [f"mechanism audit failed: {e}"],
            "risk_level": "high",
            "mechanism_mapping": {},
        }

    # AST 교차 검증: Claude 판단과 정적 분석 결과를 비교
    model_code = code_files.get("model.py", "") or code_files.get("model_py", "")
    mech_keywords = _extract_mechanism_keywords(mechanism)
    ast_result = _ast_forward_check(model_code, mech_keywords)
    result["ast_check"] = ast_result

    if ast_result["forward_found"]:
        if ast_result["ast_verified"]:
            result["evidence"].append(
                f"[AST] forward()에서 mechanism 키워드 확인: {ast_result['mechanism_hits']}"
            )
        else:
            # AST에서 미발견 but Claude says implemented → 경고
            if result.get("implemented"):
                result["evidence"].append(
                    f"[AST 경고] Claude는 구현 판단했으나 forward()에서 "
                    f"mechanism 키워드 미발견: searched={mech_keywords}, "
                    f"forward_calls={ast_result['calls_in_forward'][:10]}"
                )
                result["risk_level"] = "medium" if result["risk_level"] == "low" else result["risk_level"]
            else:
                result["missing_links"].append(
                    f"[AST] forward()에서 mechanism 미발견: {ast_result['mechanism_missing']}"
                )
    else:
        result["evidence"].append("[AST] forward() 메서드를 찾을 수 없음 — AST 검증 생략")

    return result


def _ast_forward_check(model_code: str, mechanism_keywords: list[str]) -> dict:
    """AST 정적 분석: model.py의 forward() 메서드 내 호출 그래프에서
    mechanism 키워드가 실제로 참여하는지 검증한다.

    Returns:
        {
            "forward_found": bool,
            "calls_in_forward": [str],
            "mechanism_hits": [str],
            "mechanism_missing": [str],
            "ast_verified": bool,
        }
    """
    import ast

    result = {
        "forward_found": False,
        "calls_in_forward": [],
        "mechanism_hits": [],
        "mechanism_missing": list(mechanism_keywords),
        "ast_verified": False,
    }

    if not model_code or not model_code.strip():
        return result

    try:
        tree = ast.parse(model_code)
    except SyntaxError:
        return result

    # forward() 메서드 찾기 (클래스 내부)
    forward_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "forward":
                forward_nodes.append(node)

    if not forward_nodes:
        return result

    result["forward_found"] = True

    # forward() 내부의 모든 호출 이름 수집
    calls: set[str] = set()
    for fwd in forward_nodes:
        for child in ast.walk(fwd):
            if isinstance(child, ast.Call):
                # self.xxx() → xxx
                if isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr.lower())
                # xxx() → xxx
                elif isinstance(child.func, ast.Name):
                    calls.add(child.func.id.lower())

    # 클래스 __init__에서 self.xxx = Module() 패턴의 모듈명도 수집
    init_modules: dict[str, str] = {}  # attr_name → class_name
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if (isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"
                                and isinstance(child.value, ast.Call)):
                            class_name = ""
                            if isinstance(child.value.func, ast.Name):
                                class_name = child.value.func.id
                            elif isinstance(child.value.func, ast.Attribute):
                                class_name = child.value.func.attr
                            if class_name:
                                init_modules[target.attr.lower()] = class_name.lower()

    # forward에서 호출된 self.xxx의 실제 클래스명으로 확장
    expanded_calls: set[str] = set(calls)
    for call_name in calls:
        if call_name in init_modules:
            expanded_calls.add(init_modules[call_name])

    result["calls_in_forward"] = sorted(expanded_calls)

    # mechanism 키워드 매칭
    hits = []
    missing = []
    for kw in mechanism_keywords:
        kw_lower = kw.lower().replace("-", "").replace("_", "")
        found = any(
            kw_lower in c.replace("-", "").replace("_", "")
            for c in expanded_calls
        )
        if found:
            hits.append(kw)
        else:
            missing.append(kw)

    result["mechanism_hits"] = hits
    result["mechanism_missing"] = missing
    result["ast_verified"] = len(hits) > 0

    return result


def _extract_mechanism_keywords(mechanism: str) -> list[str]:
    """mechanism 문자열에서 코드 매칭 가능한 키워드를 추출한다."""
    import re
    # 기술 용어 패턴: CamelCase, 약어, 하이픈 연결어
    tokens = re.findall(r"[A-Z][a-z]+(?:[A-Z][a-z]+)*|[A-Z]{2,}|[a-z]+(?:[-_][a-z]+)+", mechanism)
    # 일반적인 단어 필터링
    _GENERIC = {"the", "and", "for", "with", "via", "use", "using", "based", "model",
                "method", "approach", "technique", "module", "layer", "block", "network",
                "learning", "training", "deep"}
    keywords = [t for t in tokens if t.lower() not in _GENERIC and len(t) >= 3]
    # 소문자 토큰도 추가
    words = re.findall(r"\b[a-z]{4,}\b", mechanism.lower())
    for w in words:
        if w not in _GENERIC and w not in [k.lower() for k in keywords]:
            keywords.append(w)
    return keywords[:10]  # 최대 10개


def _metric_audit(spec: dict, code_files: dict) -> dict:
    """metric 키 일관성 감사: spec / code / stdout 계약 간 일치 여부 검사.

    LLM 호출 없이 순수 코드 검사로 수행한다.

    Returns:
        {
            "primary_metric_expected": str,
            "primary_metric_found": bool,
            "required_keys_expected": [...],
            "required_keys_found": [...],
            "stdout_contract_ok": bool,
            "implemented": bool
        }
    """
    ev_cfg = spec.get("evaluation_config", {})
    out_ctr = spec.get("output_contract", {})
    primary_metric = ev_cfg.get("primary_metric", "")
    required_keys = out_ctr.get("required_keys", [primary_metric] if primary_metric else [])

    # 모든 코드를 합쳐서 검색
    combined_src = "".join(code_files.values())

    # primary metric 존재 여부 (문자열 리터럴로 검색)
    primary_found = bool(
        primary_metric
        and (f'"{primary_metric}"' in combined_src or f"'{primary_metric}'" in combined_src)
    )

    # required_keys 존재 여부
    found_keys = []
    for key in required_keys:
        if f'"{key}"' in combined_src or f"'{key}'" in combined_src:
            found_keys.append(key)

    # METRICS stdout 패턴 존재 여부
    stdout_ok = "METRICS:" in combined_src

    # 전체 구현 여부: primary_metric 존재 + stdout 계약 존재
    implemented = primary_found and stdout_ok and len(found_keys) == len(required_keys)

    return {
        "primary_metric_expected": primary_metric,
        "primary_metric_found": primary_found,
        "required_keys_expected": required_keys,
        "required_keys_found": found_keys,
        "stdout_contract_ok": stdout_ok,
        "implemented": implemented,
    }


def _constraints_audit(constraints_raw: str, spec: dict, code_files: dict) -> dict:
    """constraints가 config와 아키텍처에 반영되었는지 검사.

    constraints_raw를 키워드로 파싱하고, config/코드에서 위반 여부를 확인한다.

    Returns:
        {
            "constraints_raw": str,
            "recognized_constraints": [...],
            "config_alignment": {...},
            "architecture_alignment": {...},
            "violations": [...],
            "implemented": bool
        }
    """
    if not constraints_raw or not constraints_raw.strip():
        return {
            "constraints_raw": "",
            "recognized_constraints": [],
            "config_alignment": {},
            "architecture_alignment": {},
            "violations": [],
            "implemented": True,
        }

    # 제약 조건 키워드 인식 패턴 (일반화된 매핑)
    _CONSTRAINT_PATTERNS = {
        "param_budget":       r"(\d+(?:\.\d+)?)\s*[Mm](?:illion)?\s*param",
        "single_gpu":         r"single\s*gpu|1\s*gpu|단일\s*gpu",
        "lightweight":        r"lightweight|경량|가벼운|light\s*weight|fast\s*inference|빠른\s*추론",
        "short_training":     r"short\s*train|빠른\s*학습|fast\s*train|few\s*epoch",
        "memory_friendly":    r"memory\s*friendly|메모리\s*효율|low\s*memory|memory\s*efficient",
        "no_pretrained":      r"pretrained\s*금지|no\s*pretrain|from\s*scratch|사전학습\s*없",
        "augmentation_limit": r"augmentation\s*제한|minimal\s*augmentation|no\s*augmentation",
    }

    recognized: list[str] = []
    constraints_lower = constraints_raw.lower()
    for name, pattern in _CONSTRAINT_PATTERNS.items():
        if re.search(pattern, constraints_lower, re.IGNORECASE):
            recognized.append(name)

    # param budget 추출 (숫자 + M)
    param_budget = spec.get("model_architecture", {}).get("param_budget_M")
    param_estimate = spec.get("model_architecture", {}).get("param_estimate_M")

    violations: list[str] = []
    config_alignment: dict = {}
    architecture_alignment: dict = {}

    # param budget 검사
    if param_budget is not None and param_estimate is not None:
        config_alignment["param_budget_M"] = param_budget
        config_alignment["param_estimate_M"] = param_estimate
        if float(param_estimate) > float(param_budget):
            violations.append(
                f"param_estimate ({param_estimate}M) exceeds param_budget ({param_budget}M)"
            )

    # single GPU 검사
    combined_src = "".join(code_files.values())
    if "single_gpu" in recognized:
        multi_patterns = ["DistributedDataParallel", "DataParallel", "multi_gpu", "num_nodes"]
        for pat in multi_patterns:
            if pat in combined_src:
                violations.append(f"single GPU constraint violated: '{pat}' found in code")
        architecture_alignment["single_gpu_check"] = "passed" if not any(
            pat in combined_src for pat in multi_patterns
        ) else "failed"

    # lightweight/fast inference 검사 (모델 크기 기반 간접 검사)
    if "lightweight" in recognized and param_budget is not None:
        architecture_alignment["lightweight_budget_M"] = param_budget

    # no_pretrained 검사
    if "no_pretrained" in recognized:
        pretrained_patterns = ["pretrained=True", "pretrained=true", "from_pretrained", "load_pretrained"]
        for pat in pretrained_patterns:
            if pat.lower() in combined_src.lower():
                violations.append(f"no_pretrained constraint violated: '{pat}' found in code")
        architecture_alignment["no_pretrained_check"] = "passed" if not any(
            pat.lower() in combined_src.lower() for pat in pretrained_patterns
        ) else "failed"

    implemented = len(violations) == 0

    return {
        "constraints_raw": constraints_raw,
        "recognized_constraints": recognized,
        "config_alignment": config_alignment,
        "architecture_alignment": architecture_alignment,
        "violations": violations,
        "implemented": implemented,
    }


def _save_audit_results(pkg_dir: Path, mechanism: dict, metric: dict, constraints: dict) -> None:
    """audit 결과를 artifacts/ 디렉토리에 JSON으로 저장한다."""
    artifacts_dir = pkg_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in [
        ("mechanism_audit.json", mechanism),
        ("metric_audit.json", metric),
        ("constraints_audit.json", constraints),
    ]:
        path = artifacts_dir / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    [audit] {path.name} 저장")


# ──────────────────────────────────────────────────────────
# Step 5 — Post-generation validation + 1회 repair
# ──────────────────────────────────────────────────────────

def _validate_generated_package(pkg_dir: Path) -> dict:
    """생성된 패키지의 syntax / smoke / forward 검증을 실행한다.

    Returns:
        {
            "ok": bool,           # syntax_ok AND smoke_ok
            "syntax_ok": bool,
            "smoke_ok": bool,
            "forward_ok": bool,
            "errors": [...],
            "warnings": [...]
        }
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # ── 1. Python syntax check ──────────────────────────
    syntax_ok = True
    for py_file in ["model.py", "module.py", "data.py", "train.py"]:
        fpath = pkg_dir / py_file
        if not fpath.exists():
            errors.append(f"missing file: {py_file}")
            syntax_ok = False
            continue
        try:
            py_compile.compile(str(fpath), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"syntax error in {py_file}: {e}")
            syntax_ok = False

    # ── 1b. Interface contract check (함수명/클래스명 계약) ──
    _INTERFACE_CONTRACTS = {
        "model.py": ["build_model"],
        "module.py": ["TrainingModule"],
        "data.py": ["build_dataloaders"],
    }
    for fname, required_names in _INTERFACE_CONTRACTS.items():
        fpath = pkg_dir / fname
        if fpath.exists():
            src = fpath.read_text(encoding="utf-8")
            for name in required_names:
                if name not in src:
                    errors.append(
                        f"interface contract violation: {fname} must define '{name}' "
                        f"(train.py depends on this exact name)"
                    )
                    syntax_ok = False  # hard fail

    # ── 1c. config hard access 금지 (config["key"] → config.get("key", default)) ──
    _CONFIG_HARD_ACCESS = re.compile(r"""config\s*\[\s*["'](\w+)["']\s*\]""")
    for py_file in ["train.py", "model.py", "module.py", "data.py"]:
        fpath = pkg_dir / py_file
        if not fpath.exists():
            continue
        src = fpath.read_text(encoding="utf-8")
        hard_keys = _CONFIG_HARD_ACCESS.findall(src)
        if hard_keys:
            errors.append(
                f'{py_file}: config hard access 금지 — config["{k}"] → '
                f'config.get("{k}", default) 로 변경 필요: {hard_keys}'
            )
            syntax_ok = False

    # ── 2. Config / file sanity ─────────────────────────
    for required in ["configs/default.yaml", "configs/fast.yaml", "scripts/smoke_test.py"]:
        if not (pkg_dir / required).exists():
            errors.append(f"missing required file: {required}")

    # ── 2b. default.yaml flat 구조 + 필수 키 검증 ──────
    _REQUIRED_CONFIG_KEYS = {
        "seed", "epochs", "batch_size", "lr", "data_dir", "num_workers", "primary_metric",
    }
    yaml_path = pkg_dir / "configs" / "default.yaml"
    if yaml_path.exists():
        import yaml as _yaml
        try:
            cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            # nested 구조 감지 (value가 dict이면 nested)
            nested_keys = [k for k, v in cfg.items() if isinstance(v, dict)]
            if nested_keys:
                errors.append(
                    f"default.yaml: flat 구조 필수 — nested 키 발견: {nested_keys}. "
                    f"예: seed: 42 (O) / training: {{seed: 42}} (X)"
                )
                syntax_ok = False
            # 필수 키 누락 검사
            missing = _REQUIRED_CONFIG_KEYS - set(cfg.keys())
            if missing:
                errors.append(f"default.yaml: 필수 키 누락: {sorted(missing)}")
                syntax_ok = False
        except Exception as e:
            errors.append(f"default.yaml parse error: {e}")

    # ── 2c. data_path 사용 검증 (명시된 경로가 있으면 data.py에서 참조 필수) ──
    if yaml_path.exists():
        try:
            cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            data_dir = cfg.get("data_dir", "")
            if data_dir and data_dir != "data/" and os.path.isabs(data_dir):
                data_src = (pkg_dir / "data.py").read_text(encoding="utf-8") if (pkg_dir / "data.py").exists() else ""
                if data_dir not in data_src and "config.get" not in data_src:
                    warnings.append(
                        f"data.py: data_dir '{data_dir}'가 코드에서 참조되지 않음 — "
                        f"합성 데이터 대신 실제 데이터를 로드하는지 확인 필요"
                    )
        except Exception:
            pass

    # ── 3. smoke test ───────────────────────────────────
    smoke_ok = False
    if syntax_ok:
        cmd = [sys.executable, "scripts/smoke_test.py", "--config", "configs/fast.yaml"]
        try:
            _proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=str(pkg_dir), start_new_session=True,
            )
            try:
                _stdout, _stderr = _proc.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                import os as _os, signal as _sig
                try:
                    _os.killpg(_proc.pid, _sig.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                _proc.wait()
                raise
            proc = subprocess.CompletedProcess(cmd, _proc.returncode, _stdout, _stderr)
            smoke_ok = proc.returncode == 0
            if not smoke_ok:
                errors.append(f"smoke_test failed: {proc.stderr[-400:]}")
        except subprocess.TimeoutExpired:
            errors.append("smoke_test timeout (180s)")
        except Exception as e:
            errors.append(f"smoke_test exception: {e}")
    else:
        warnings.append("smoke_test skipped due to syntax errors")

    # ── 4. forward test (non-blocking) ─────────────────
    forward_ok = True
    test_path = pkg_dir / "tests" / "test_forward.py"
    if test_path.exists() and syntax_ok:
        cmd = [sys.executable, "-m", "pytest", str(test_path), "-q", "--tb=short"]
        try:
            _proc2 = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=str(pkg_dir), start_new_session=True,
            )
            try:
                _stdout2, _stderr2 = _proc2.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                import os as _os, signal as _sig
                try:
                    _os.killpg(_proc2.pid, _sig.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                _proc2.wait()
                raise
            proc = subprocess.CompletedProcess(cmd, _proc2.returncode, _stdout2, _stderr2)
            forward_ok = proc.returncode == 0
            if not forward_ok:
                warnings.append(f"test_forward failed (non-blocking): {proc.stdout[-300:]}")
        except Exception as e:
            warnings.append(f"test_forward exception (non-blocking): {e}")
    else:
        warnings.append("tests/test_forward.py not found — skipped")

    # ── 5. metric key contract check ────────────────────
    # spec.output_contract.required_keys 가 module.py 또는 train.py에 존재하는지 확인
    metric_contract_ok = True
    spec_path = pkg_dir / "experiment_spec.json"
    if spec_path.exists() and syntax_ok:
        try:
            spec_data   = json.loads(spec_path.read_text(encoding="utf-8"))
            req_keys    = spec_data.get("output_contract", {}).get("required_keys", [])
            module_src  = (pkg_dir / "module.py").read_text(encoding="utf-8") if (pkg_dir / "module.py").exists() else ""
            train_src   = (pkg_dir / "train.py").read_text(encoding="utf-8") if (pkg_dir / "train.py").exists() else ""
            combined_src = module_src + train_src
            missing_keys = [k for k in req_keys if f'"{k}"' not in combined_src and f"'{k}'" not in combined_src]
            if missing_keys:
                warnings.append(f"metric key contract: required_keys {missing_keys} not found in module.py/train.py")
                metric_contract_ok = False
        except Exception as e:
            warnings.append(f"metric contract check error: {e}")
    else:
        warnings.append("metric contract check skipped (no experiment_spec.json or syntax errors)")

    # ── 6. output contract check ─────────────────────
    # METRICS:{...} stdout 패턴이 train.py 또는 module.py에 존재하는지 확인
    output_contract_ok = False
    train_src_check = (pkg_dir / "train.py").read_text(encoding="utf-8") if (pkg_dir / "train.py").exists() else ""
    module_src_check = (pkg_dir / "module.py").read_text(encoding="utf-8") if (pkg_dir / "module.py").exists() else ""
    if "METRICS:" in train_src_check or "METRICS:" in module_src_check:
        output_contract_ok = True
    else:
        warnings.append("output contract: METRICS: stdout pattern not found in train.py or module.py")

    # ── 7. Hypothesis implementation audit ─────────────────
    # hard gate: 실패 시 패키지 생성 차단 (repair 1회 후에도 실패 시)
    # soft gate: warning만 기록 (차단하지 않음)
    mechanism_ok   = True
    metric_ok      = True
    constraints_ok = True
    hypothesis_implementation_ok = True
    hard_failures: list[str] = []
    soft_warnings: list[str] = []

    mechanism_result    = {}
    metric_result       = {}
    constraints_result  = {}

    if spec_path.exists() and syntax_ok:
        try:
            spec_data = json.loads(spec_path.read_text(encoding="utf-8"))
            code_files = _load_code_files(pkg_dir)
            hyp_contract = spec_data.get("hypothesis_contract", {})

            # ── mechanism audit (Claude 기반) ──
            mechanism_text = hyp_contract.get("mechanism", "")
            arch_summary = spec_data.get("model_architecture", {}).get("description", "")
            mechanism_result = _mechanism_audit(mechanism_text, code_files, arch_summary)
            mechanism_ok = mechanism_result.get("implemented", False)

            # hard gate: mechanism이 명시되어 있는데 미구현이면 차단
            mechanism_required = bool(mechanism_text and mechanism_text.strip())
            if not mechanism_ok and mechanism_required:
                hard_failures.append(
                    f"mechanism not implemented: "
                    f"missing={mechanism_result.get('missing_links', [])}"
                )
            elif not mechanism_ok:
                # mechanism 필드가 비어있으면 soft warning만
                soft_warnings.append("mechanism audit: mechanism 필드 비어있음 (감사 생략)")

            # ── metric audit (순수 코드 검사) ──
            metric_result = _metric_audit(spec_data, code_files)
            metric_ok = metric_result.get("implemented", False)

            # hard gate: metric 미구현은 실험 자체가 성립하지 않음
            if not metric_ok:
                hard_failures.append(
                    f"metric not implemented: "
                    f"primary={metric_result.get('primary_metric_expected')}, "
                    f"found={metric_result.get('primary_metric_found')}, "
                    f"stdout_ok={metric_result.get('stdout_contract_ok')}, "
                    f"missing_keys={[k for k in metric_result.get('required_keys_expected', []) if k not in metric_result.get('required_keys_found', [])]}"
                )

            # ── constraints audit ──
            constraints_result = _constraints_audit(
                hyp_contract.get("constraints_raw", ""), spec_data, code_files
            )
            constraints_ok = constraints_result.get("implemented", False)
            violations = constraints_result.get("violations", [])

            if violations:
                # hard gate: 명백한 위반 (param budget 초과, pretrained 금지 위반 등)
                hard_failures.append(
                    f"constraints violated: {violations}"
                )
            elif not constraints_ok:
                # soft warning: 위반은 없지만 인식된 제약이 불완전
                soft_warnings.append(
                    f"constraints audit: recognized={constraints_result.get('recognized_constraints', [])}"
                )

            hypothesis_implementation_ok = mechanism_ok and metric_ok and constraints_ok

            # audit 결과 저장
            _save_audit_results(pkg_dir, mechanism_result, metric_result, constraints_result)

        except Exception as e:
            warnings.append(f"hypothesis implementation audit error: {e}")

    # hard audit 통과 여부
    # mechanism_required가 아니면 mechanism은 hard gate에서 제외
    hard_mechanism_ok = mechanism_ok if (mechanism_result and bool(
        mechanism_result.get("mechanism_mapping") or
        hyp_contract.get("mechanism", "").strip() if 'hyp_contract' in dir() else True
    )) else True
    hard_audit_ok = metric_ok and constraints_ok and hard_mechanism_ok

    # soft warning을 warnings에 추가
    warnings.extend(soft_warnings)

    # ── 8. Task-family contract audit (hard gate) ──────
    task_family_ok = True
    family_audit_result = {}
    if spec_path.exists() and syntax_ok:
        try:
            from lab.task_families import run_family_contract_tests
            spec_data = json.loads(spec_path.read_text(encoding="utf-8"))
            tf = spec_data.get("task_family", "classification")
            code_files = {}
            for f in ["model.py", "module.py", "data.py"]:
                fp = pkg_dir / f
                if fp.exists():
                    code_files[f] = fp.read_text(encoding="utf-8")
            yaml_fp = pkg_dir / "configs" / "default.yaml"
            if yaml_fp.exists():
                code_files["default.yaml"] = yaml_fp.read_text(encoding="utf-8")
            family_audit_result = run_family_contract_tests(tf, code_files, spec_data)
            task_family_ok = family_audit_result.get("task_family_contract_ok", True)
            if not task_family_ok:
                for fail_msg in family_audit_result.get("failed", []):
                    hard_failures.append(f"[family:{tf}] {fail_msg}")
                print(f"    [family audit] {tf}: FAIL — {family_audit_result.get('failed', [])}")
            else:
                print(f"    [family audit] {tf}: PASS ({len(family_audit_result.get('passed', []))} tests)")
        except Exception as e:
            warnings.append(f"task_family audit error: {e}")

    # 최종 ok: syntax + smoke + hard audit + family audit
    ok = syntax_ok and smoke_ok and hard_audit_ok and task_family_ok

    return {
        "ok":                  ok,
        "syntax_ok":           syntax_ok,
        "smoke_ok":            smoke_ok,
        "forward_ok":          forward_ok,
        "metric_contract_ok":  metric_contract_ok,
        "output_contract_ok":  output_contract_ok,
        "hypothesis_implementation_ok": hypothesis_implementation_ok,
        "mechanism_ok":        mechanism_ok,
        "metric_ok":           metric_ok,
        "constraints_ok":      constraints_ok,
        "hard_audit_ok":       hard_audit_ok,
        "task_family_ok":      task_family_ok,
        "family_audit":        family_audit_result,
        "hard_failures":       hard_failures,
        "errors":              errors,
        "warnings":            warnings,
    }


def _write_validation_report(pkg_dir: Path, report: dict, phase: str = "initial") -> Path:
    """proposals/validation_{phase}_{ts_ms}.json 으로 저장하고 경로 반환.

    Args:
        phase: "initial" (첫 검증) 또는 "repaired" (repair 후 재검증).
               phase 레이블 + 밀리초 타임스탬프로 파일명 충돌을 방지한다.
    """
    proposals_dir = pkg_dir / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)  # parents=True: pkg_dir가 삭제된 경우 재생성
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")   # %f = 마이크로초 6자리
    path = proposals_dir / f"validation_{phase}_{ts}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [validation] {path.name} 저장")
    return path


def _claude_repair_from_validation(
    pkg_dir: Path,
    validation_report: dict,
    spec: dict,
    hypothesis: dict,
) -> dict:
    """검증 실패 시 Claude가 1회 수정 패스를 수행한다.

    - train.py 및 METRICS stdout 계약은 절대 수정 금지
    - 오류를 유발하는 최소 부분만 수정
    """
    errors  = list(validation_report.get("errors", []))
    # hard audit 실패 사유를 repair context에 구조적으로 포함
    hard_failures = validation_report.get("hard_failures", [])
    if hard_failures:
        errors.append("=== Hypothesis Implementation Hard Failures (반드시 수정) ===")
        errors.extend(hard_failures)
    # soft warning도 참고용으로 포함
    audit_warnings = [w for w in validation_report.get("warnings", [])
                      if any(k in w for k in ("mechanism", "metric", "constraints"))]
    if audit_warnings:
        errors.extend([f"[audit warning] {w}" for w in audit_warnings])
    hyp     = hypothesis.get("hypothesis", {})

    # 현재 파일 읽기
    file_contents: dict[str, str] = {}
    for key, rel in [
        ("model_py",     "model.py"),
        ("module_py",    "module.py"),
        ("data_py",      "data.py"),
        ("default_yaml", "configs/default.yaml"),
    ]:
        fpath = pkg_dir / rel
        file_contents[key] = fpath.read_text(encoding="utf-8") if fpath.exists() else ""

    prompt = f"""당신은 실험 패키지 수리 담당자(Claude 역할)입니다.
아래 검증 실패 오류를 분석하고 최소한의 수정으로 오류를 해결하세요.

## 검증 오류
{json.dumps(errors, ensure_ascii=False, indent=2)}

## 수정 가능 파일 (train.py / METRICS stdout 계약은 절대 수정 금지)
- model.py
- module.py
- data.py
- configs/default.yaml

## 현재 model.py
```python
{file_contents.get('model_py', '')[:3000]}
```

## 현재 module.py
```python
{file_contents.get('module_py', '')[:2000]}
```

## 현재 data.py
```python
{file_contents.get('data_py', '')[:2000]}
```

## 현재 default.yaml
```yaml
{file_contents.get('default_yaml', '')[:1000]}
```

## 가설 (변경 금지 — 구현 방향 참고용)
{hyp.get('statement_kr', hyp.get('statement', ''))}

## 지시사항
1. 오류를 유발하는 최소 부분만 수정
2. train.py 수정 금지
3. METRICS:{{...}} stdout 출력 계약 보존
4. 불필요한 리팩토링 금지

아래 JSON으로만 출력:
{{
  "model_py": "수정된 model.py 전체 (변경 없으면 빈 문자열)",
  "module_py": "수정된 module.py 전체 (변경 없으면 빈 문자열)",
  "data_py": "수정된 data.py 전체 (변경 없으면 빈 문자열)",
  "default_yaml": "수정된 default.yaml 전체 (변경 없으면 빈 문자열)",
  "repair_summary": "수정 내용 요약"
}}"""

    print("  [Step 5 / Claude] 검증 실패 — 1회 수정 패스...")
    result = None
    for _attempt in range(3):
        try:
            raw = query_claude(prompt, model="claude-sonnet-4-20250514")
            if not raw or not raw.strip():
                print(f"    [Repair] 빈 응답 (시도 {_attempt+1}/3), 재시도...")
                continue
            result = parse_json(raw)
            break
        except Exception as e:
            print(f"    [Repair] 파싱 실패 (시도 {_attempt+1}/3): {e}")
            if _attempt == 2:
                raise
    if result is None:
        raise RuntimeError("Repair: 3회 시도 후 빈 응답")

    # 빈 문자열 = 변경 없음 → 기존 파일 유지
    for key in ("model_py", "module_py", "data_py", "default_yaml"):
        if not result.get(key):
            result[key] = file_contents.get(key, "")

    return result


# ──────────────────────────────────────────────────────────
# 생성 파일 저장
# ──────────────────────────────────────────────────────────

def _write_package_files(pkg_dir: Path, files: dict) -> None:
    """model_py / module_py / data_py / default_yaml → 패키지에 저장."""
    file_map = {
        "model_py":     "model.py",
        "module_py":    "module.py",
        "data_py":      "data.py",
        "default_yaml": "configs/default.yaml",
    }
    for key, rel_path in file_map.items():
        content = files.get(key, "")
        if not content:
            print(f"    [경고] {rel_path} 비어있음 — template 유지")
            continue
        dest = pkg_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        print(f"    [저장] {dest.relative_to(pkg_dir.parent)}")


# ──────────────────────────────────────────────────────────
# 공통 후처리 (proposal archive + spec + readme 저장)
# ──────────────────────────────────────────────────────────

def _finalize_package(
    pkg_dir: Path,
    merged: dict,
    gpt_path: Path,
    gem_path: Path,
    spec: dict,
    topic_file: str,
    version: int,
    slug: str,
) -> dict:
    """Merge 완료 후 proposal archive, experiment_spec, README 갱신."""
    merge_log = merged.get("merge_log", [])
    _save_proposal(
        pkg_dir, "merge_log",
        {"merge_log": merge_log, "created_at": datetime.now().isoformat()},
    )

    accepted_count = sum(1 for m in merge_log if m.get("accepted"))
    print(f"    Merge 완료: {accepted_count}/{len(merge_log)}개 제안 반영")

    gpt_accepted = any(m.get("accepted") and m.get("source") == "GPT" for m in merge_log)
    gem_accepted = any(m.get("accepted") and m.get("source") == "Gemini" for m in merge_log)
    _archive_proposal(gpt_path, gpt_accepted)
    _archive_proposal(gem_path, gem_accepted)

    spec["model_architecture"]["param_estimate_M"] = merged.get("param_estimate_M")
    spec["model_architecture"]["description"]      = merged.get("description", "")
    spec_path = pkg_dir / "experiment_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_path = pkg_dir / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        for placeholder, value in [
            ("{TOPIC_SLUG}", slug), ("{N}", str(version)),
            ("{hypothesis_summary}", merged.get("architecture_summary", "")),
            ("{spec_id}", spec["spec_id"]),
            ("{hypothesis_id}", spec["hypothesis_id"]),
            ("{primary_metric}", spec["evaluation_config"]["primary_metric"]),
            ("{target_value}", str(spec["evaluation_config"]["target_value"])),
        ]:
            readme = readme.replace(placeholder, value)
        readme_path.write_text(readme, encoding="utf-8")

    rpt_dir = _get_reports_dir(slug)
    rpt_dir.mkdir(parents=True, exist_ok=True)
    (rpt_dir / f"experiment_spec_v{version}.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "pkg_dir":               str(pkg_dir),
        "spec_id":               spec["spec_id"],
        "hypothesis_id":         spec["hypothesis_id"],
        "experiment_version":    version,
        "description":           merged.get("description", ""),
        "architecture_summary":  merged.get("architecture_summary", ""),
        "param_estimate_M":      merged.get("param_estimate_M"),
        "primary_metric":        spec["evaluation_config"]["primary_metric"],
        "target_value":          spec["evaluation_config"]["target_value"],
        "merge_log":             merge_log,
        "timestamp":             datetime.now().isoformat(),
    }


# ──────────────────────────────────────────────────────────
# 패키지 생성 메인 함수
# ──────────────────────────────────────────────────────────

def generate_experiment_package(
    topic_file: str,
    hypothesis_file: str,
    code_analysis_file: str,
    version: int = 1,
    revised_from: str | None = None,
    revision_path: str | None = None,
    improvement_hints: str = "",
    accepted_gpt_patches: list | None = None,
    accepted_gpt_indexes: list[int] | None = None,
    gemini_review_ctx: dict | None = None,
    result_summary_ctx: dict | None = None,
) -> dict:
    """
    Multi-model Proposal-Review-Merge 흐름으로 실험 패키지를 생성한다.

    Path A revision 모드 (revised_from 지정 + revision_path="A"):
      - 이전 패키지를 복사 후 Claude가 최소 diff revision
      - 전체 template 재생성 대신 변경 필요 파일만 수정

    일반 생성 모드 (revised_from 없음 또는 Path B/C):
      - experiments/template/ 복사 후 Claude가 처음부터 생성
    """
    topic         = json.loads(Path(topic_file).read_text(encoding="utf-8"))
    hypothesis    = json.loads(Path(hypothesis_file).read_text(encoding="utf-8"))
    code_analysis = json.loads(Path(code_analysis_file).read_text(encoding="utf-8"))

    from lab.config import run_dir, reports_dir as _get_reports_dir
    inp     = topic.get("input", {})
    slug    = _topic_slug(inp.get("topic", "research"))
    pkg_dir = run_dir(slug, version)

    is_path_a_revision = bool(revised_from and revision_path == "A")

    print(f"\n{'─'*60}")
    print(f"  [7단계] 실험 패키지 생성: {pkg_dir}")
    mode_label = "Path A revision" if is_path_a_revision else "template 기반 최초 생성"
    print(f"  모드: {mode_label}")
    print(f"  proposal-review-merge: Claude → GPT → Gemini → Claude")
    print(f"{'─'*60}")

    spec = _build_experiment_spec(
        topic, hypothesis, code_analysis, version, revised_from, revision_path
    )

    # ── 패키지 초기화 ─────────────────────────────────────
    if is_path_a_revision:
        previous_pkg = Path(revised_from)
        _copy_previous_package(previous_pkg, pkg_dir)
        prev_ctx = _load_previous_package_context(previous_pkg)
        if result_summary_ctx:
            prev_ctx["result_summary"] = result_summary_ctx

        # accepted_gpt_patches + accepted_gpt_indexes → 구조화된 patch context 생성
        patch_ctx = None
        if accepted_gpt_patches:
            indexes = accepted_gpt_indexes if accepted_gpt_indexes is not None \
                      else list(range(len(accepted_gpt_patches)))
            patch_ctx = _prepare_accepted_patch_context(accepted_gpt_patches, indexes)
            print(f"    [Path A] 구조화된 패치 context: {patch_ctx['accepted_count']}개 "
                  f"→ 대상 파일: {patch_ctx['target_files']}")

        base = _claude_generate_revision_patch(
            prev_ctx, hypothesis, spec,
            improvement_hints = improvement_hints,
            patch_context     = patch_ctx,
            gemini_review     = gemini_review_ctx,
        )
    else:
        # template 복사 → Claude 기반 코드 생성
        if pkg_dir.exists():
            shutil.rmtree(pkg_dir)
        shutil.copytree(TEMPLATE_DIR, pkg_dir)
        (pkg_dir / "proposals").mkdir(exist_ok=True)
        for sub in ["checkpoints", "logs", "metrics"]:
            (pkg_dir / "artifacts" / sub).mkdir(parents=True, exist_ok=True)
        print(f"  [template 복사] → {pkg_dir}")

        base = _claude_generate_base(
            topic, hypothesis, code_analysis, spec, improvement_hints
        )

    _write_package_files(pkg_dir, base)
    print(f"    기반 코드 완료: {base.get('architecture_summary', '')}")

    # ── Step 2: GPT 패치 제안 ─────────────────────────────
    gpt_proposal = _gpt_propose_patches(base, spec, hypothesis)
    gpt_path     = _save_proposal(pkg_dir, "gpt_patch", gpt_proposal)

    # ── Step 3-A: Gemini 설계 리뷰 ────────────────────────
    gemini_review = _gemini_design_review(base, spec, hypothesis)
    gem_path      = _save_proposal(pkg_dir, "gemini_review", gemini_review)

    # ── Step 3-B/C/D: Patch Ballot + 2-of-3 Merge Decision ──
    gpt_patches = gpt_proposal.get("patches", [])
    merge_decisions: list[dict] | None = None

    if gpt_patches:
        # 코드 파일 로드 (ballot 평가에 필요)
        code_files = _load_code_files(pkg_dir)

        # Step 3-B: Gemini 패치별 투표
        gemini_ballot = _gemini_patch_ballot(gpt_patches, spec, hypothesis, code_files)
        _save_proposal(pkg_dir, "gemini_patch_ballot", {
            "ballots": gemini_ballot,
            "patch_count": len(gpt_patches),
            "created_at": datetime.now().isoformat(),
        })

        # Step 3-C: Claude 패치별 독립 판정
        claude_ballot = _claude_patch_ballot(gpt_patches, spec, hypothesis, code_files)
        _save_proposal(pkg_dir, "claude_patch_ballot", {
            "ballots": claude_ballot,
            "patch_count": len(gpt_patches),
            "created_at": datetime.now().isoformat(),
        })

        # Step 3-D: 2-of-3 합의 결정
        merge_decisions = _compute_patch_decisions(gpt_patches, gemini_ballot, claude_ballot)
        _save_proposal(pkg_dir, "merge_decision", {
            "decisions": merge_decisions,
            "patch_count": len(gpt_patches),
            "accepted": sum(1 for d in merge_decisions if d["final_decision"] == "accept"),
            "rejected": sum(1 for d in merge_decisions if d["final_decision"] == "reject"),
            "ambiguous": sum(1 for d in merge_decisions if d["final_decision"] == "ambiguous"),
            "created_at": datetime.now().isoformat(),
        })

    # ── Step 4: Claude Merge (합의 결과 집행) ──────────────
    merged = _claude_merge(
        base, gpt_proposal, gemini_review, spec, hypothesis,
        merge_decisions=merge_decisions,
    )
    _write_package_files(pkg_dir, merged)

    # ── Step 5: Post-generation validation + 1회 repair ──
    print("  [Step 5 / Validation] 생성 패키지 검증...")
    val_report = _validate_generated_package(pkg_dir)
    _write_validation_report(pkg_dir, val_report, phase="initial")

    # final_val_report: repair 전후 중 최종 유효 상태를 추적
    final_val_report = {**val_report, "repaired": False}

    if not val_report["ok"]:
        print(f"    [경고] 검증 실패: {val_report['errors']}")
        repaired = _claude_repair_from_validation(pkg_dir, val_report, spec, hypothesis)
        _write_package_files(pkg_dir, repaired)
        print(f"    수정 완료: {repaired.get('repair_summary', '')}")
        # 1회 재검증 (추가 반복 없음) — phase="repaired"로 충돌 방지
        val_report2 = _validate_generated_package(pkg_dir)
        _write_validation_report(pkg_dir, val_report2, phase="repaired")
        # repair 이후의 결과가 최종 상태
        final_val_report = {**val_report2, "repaired": True}
        if not val_report2["ok"]:
            print(f"    [오류] 재검증 실패: {val_report2['errors']}")
        else:
            print("    [검증] 수정 후 검증 통과 ✅")
    else:
        print("  [검증] 통과 ✅")

    # ── Finalization gate: 최종 validation 통과 시에만 승인 ──────
    if not final_val_report["ok"]:
        # 구조적 실패 사유 구성
        failure_detail = {
            "syntax":      not final_val_report.get("syntax_ok", True),
            "smoke":       not final_val_report.get("smoke_ok", True),
            "mechanism":   not final_val_report.get("mechanism_ok", True),
            "metric":      not final_val_report.get("metric_ok", True),
            "constraints": not final_val_report.get("constraints_ok", True),
            "hard_audit":  not final_val_report.get("hard_audit_ok", True),
            "repair_attempted": final_val_report.get("repaired", False),
        }
        blocked_reason = (
            "post-generation validation failed after repair"
            if final_val_report.get("repaired")
            else "post-generation validation failed"
        )
        blocked_msg = "Package generation blocked: final validation failed."
        print(f"\n  ❌ {blocked_msg}")
        print(f"     errors: {final_val_report.get('errors', [])}")
        hard_f = final_val_report.get("hard_failures", [])
        if hard_f:
            print(f"     hard_failures: {hard_f}")
        print(f"     failure_detail: {failure_detail}")
        print(f"     (artifacts retained for debugging: {pkg_dir})")
        return {
            "pkg_dir":            str(pkg_dir),
            "spec_id":            spec["spec_id"],
            "hypothesis_id":      spec["hypothesis_id"],
            "experiment_version": version,
            "success":            False,
            "finalized":          False,
            "validation_failed":  True,
            "blocked_reason":     blocked_reason,
            "failure_detail":     failure_detail,
            "hard_failures":      hard_f,
            "validation":         final_val_report,
            "timestamp":          datetime.now().isoformat(),
        }

    # ── 후처리: proposal archive + spec + readme (검증 통과한 경우만) ──
    result = _finalize_package(
        pkg_dir, merged, gpt_path, gem_path, spec, topic_file, version, slug
    )
    result["validation"]        = final_val_report
    result["success"]           = True
    result["finalized"]         = True
    result["validation_failed"] = False
    result["blocked_reason"]    = None

    print(f"\n  ✅ 패키지 생성 완료: {pkg_dir}")
    print(f"     아키텍처: {result['architecture_summary']}")
    print(f"     예상 파라미터: {result['param_estimate_M']}M")
    return result


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Fabric 실험 패키지 생성 (Multi-model)")
    parser.add_argument("--topic-file",        required=True)
    parser.add_argument("--hypothesis-file",   required=True)
    parser.add_argument("--code-file",         required=True)
    parser.add_argument("--version",           type=int, default=1)
    parser.add_argument("--revised-from",      default=None,
                        help="Path A 시 이전 패키지 경로 (예: experiments/{slug}/runs/v1)")
    parser.add_argument("--revision-path",     default=None, choices=["A", "B", "C"])
    parser.add_argument("--improvement-hints", default="")
    args = parser.parse_args()

    result = generate_experiment_package(
        topic_file         = args.topic_file,
        hypothesis_file    = args.hypothesis_file,
        code_analysis_file = args.code_file,
        version            = args.version,
        revised_from       = args.revised_from,
        revision_path      = args.revision_path,
        improvement_hints  = args.improvement_hints,
    )

    if result.get("success"):
        print(f"\n{'='*60}")
        print(f"  ✅ 패키지 생성 성공")
        print(f"  패키지: {result['pkg_dir']}")
        print(f"  실행: python {result['pkg_dir']}/train.py "
              f"--config {result['pkg_dir']}/configs/default.yaml")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"  ❌ 패키지 생성 차단 (validation gate)")
        print(f"  사유: {result.get('blocked_reason', 'unknown')}")
        print(f"  디버그 경로: {result['pkg_dir']}")
        print(f"{'='*60}")
        sys.exit(1)
