"""
Stage 7: PyTorch Fabric 실험 패키지 생성 (Multi-model Proposal-Review-Merge)

흐름:
  1. Claude   → 기반 코드 생성 (model.py / module.py / data.py / default.yaml)
  2. GPT/Codex→ 코드 패치 제안 (proposals/gpt_patch_*.json)
  3. Gemini   → 설계 리뷰     (proposals/gemini_review_*.json)
  4. Claude   → 제안 검토 + Merge Checklist 적용 → 최종 파일 확정
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
    proposals_dir.mkdir(exist_ok=True)
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
        "ablations": [],
        "baselines": [
            {"name": b.strip(), "source": "literature"}
            for b in code_analysis.get("recommended_baseline", "").split(",")
            if b.strip()
        ],
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

    prompt = f"""당신은 PyTorch Fabric 전문 딥러닝 엔지니어이며, 이 실험 패키지의 **유일한 코드 작성자**입니다.
GPT/Codex는 이후 패치만 제안하고, Gemini는 설계만 리뷰하며, 최종 merge도 당신이 수행합니다.
코드 생성 결정은 아래 experiment_spec을 최우선 계약으로 따르세요.

## [최우선] 구현 계약 (experiment_spec)
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
1. model.py: `build_model(config) -> nn.Module` — spec.model_architecture 구현, 모든 크기는 config에서 읽기
2. module.py: `TrainingModule(model, config)` — train_epoch/val_epoch 포함, val에 spec.required_keys 전부 포함
3. data.py: `build_dataloaders(config)` — 합성 더미 데이터 fallback 포함
4. default.yaml: spec.training_config 기반 (모든 하이퍼파라미터 포함)
5. METRICS stdout 계약: print(f"METRICS:{{json.dumps({{...}})}}") — required_keys 모두 출력

아래 JSON으로만 출력 (코드 블록 없이):
{{
  "model_py": "...", "module_py": "...", "data_py": "...", "default_yaml": "...",
  "description": "...", "architecture_summary": "...", "param_estimate_M": 0.0
}}"""

    print("  [Step 1 / Claude] 기반 코드 생성...")
    return parse_json(query_claude(prompt))


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
    # 중첩된 버전 폴더(v1, v2, ...), __pycache__, data/ 캐시를 제외하고 복사
    shutil.copytree(
        previous_pkg, new_pkg,
        ignore=shutil.ignore_patterns("v[0-9]*", "__pycache__", "data"),
    )

    # ── 아티팩트 디렉토리 실제 초기화 ──────────────────────
    # artifacts/ 전체를 삭제 후 빈 서브디렉토리로 재생성
    artifacts_dir = new_pkg / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    for sub in ["checkpoints", "logs", "metrics"]:
        (artifacts_dir / sub).mkdir(parents=True)

    # 패키지 루트 레벨의 checkpoints/, logs/ 디렉토리도 초기화 (template 변형 대응)
    for extra_dir in ["checkpoints", "logs"]:
        d = new_pkg / extra_dir
        if d.exists() and d.is_dir():
            shutil.rmtree(d)
            d.mkdir()

    # ── proposals 폴더: 이전 제안에 prefix 추가 ───────────
    proposals_dir = new_pkg / "proposals"
    if proposals_dir.exists():
        for old_proposal in proposals_dir.glob("*.json"):
            if not old_proposal.name.startswith("prev_"):
                old_proposal.rename(old_proposal.with_name("prev_" + old_proposal.name))

    print(f"  [Path A] 이전 패키지 복사: {previous_pkg.name} → {new_pkg.name}"
          f" (artifacts 초기화 완료)")


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
    result = parse_json(query_claude(prompt))

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

Review ONLY the following dimensions:
1. mechanism_check: does the architecture actually implement the hypothesis mechanism?
2. spec_alignment: does the design match spec.model_architecture fields?
3. missing_components: what critical components are absent?
4. unnecessary_complexity: what adds complexity without serving the hypothesis?
5. bottlenecks: where is the most likely experimental failure point?

Return JSON only:
{{
  "verdict": "accept_as_is | accept_with_patch | revise_experiment",
  "hypothesis_alignment_score": 0.0,
  "mechanism_check": "does the architecture implement the key mechanism? explain",
  "spec_alignment": "does the design match the spec fields? explain",
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
# Step 4 — Claude: Merge
# ──────────────────────────────────────────────────────────

def _claude_merge(
    generated: dict,
    gpt_proposal: dict,
    gemini_review: dict,
    spec: dict,
    hypothesis: dict,
) -> dict:
    """Claude가 GPT 패치와 Gemini 리뷰를 검토하고 최종 파일을 확정한다."""
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

    prompt = f"""당신은 실험 패키지의 최종 통합자이자 **유일한 코드 작성자(Claude 역할)**입니다.
GPT/Codex 패치와 Gemini 설계 리뷰를 아래 우선순위 체크리스트로 판단하여 최종 파일을 확정하세요.
GPT/Codex나 Gemini의 코드를 그대로 복사하지 말고, 판단 후 당신이 직접 반영 여부를 결정하세요.

## [최우선] 구현 계약 (experiment_spec — 모든 결정의 기준)
- spec_id: {spec['spec_id']}
- model_architecture: name={arch['name']}, key_components={arch.get('key_components', [])}, param_budget_M≤{param_budget}M
- evaluation_config: {primary} ≥ {ev_cfg['target_value']}, secondary={ev_cfg.get('secondary_metrics', [])}
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

## GPT/Codex 패치 제안 ({len(gpt_patches)}개) — 패치만 제안, 전체 재작성 아님
{json.dumps(gpt_patches, ensure_ascii=False, indent=2)}

## Gemini 설계 리뷰 — 설계 비평만, 코드 제안 아님
- verdict: {gemini_review.get('verdict')}
- mechanism_check: {gemini_review.get('mechanism_check', '')}
- spec_alignment: {gemini_review.get('spec_alignment', '')}
- alignment_score: {gemini_review.get('hypothesis_alignment_score')}
- critical/major issues: {json.dumps(gemini_issues, ensure_ascii=False)}
- missing_components: {gemini_review.get('missing_components', [])}
- bottlenecks: {gemini_review.get('bottlenecks', [])}

## Merge 우선순위 체크리스트 (이 순서로 판단)
1. [spec 호환성] 패치/수정이 spec.model_architecture / training_config / output_contract를 위반하지 않는가?
2. [가설 정합성] 패치/수정이 가설 메커니즘을 구현하거나 보강하는가?
3. [머신 검증 가능성] 적용 후 syntax compile + smoke_test 통과 가능한가?
4. [output contract 보존] METRICS stdout 계약 및 required_keys가 유지되는가?
5. [최소 변경] complexity_delta_loc ≤ 50이고 불필요한 리팩토링이 없는가?
6. [선택적 개선] 위 5가지 통과 후에만 optional 개선 반영

## 지시사항
- train.py 절대 수정 금지
- METRICS stdout 계약 보존 (required_keys 이름 변경 금지)
- 패치별 accepted/rejected 이유를 spec_field와 함께 명시

아래 JSON으로만 출력:
{{
  "model_py": "최종 model.py 전체",
  "module_py": "최종 module.py 전체",
  "data_py": "최종 data.py 전체",
  "default_yaml": "최종 default.yaml 전체",
  "merge_log": [
    {{"source": "GPT|Gemini", "item": "...", "spec_field": "...", "accepted": true, "reason": "..."}}
  ],
  "description": "최종 모델 설명",
  "architecture_summary": "한 줄 요약",
  "param_estimate_M": 0.0
}}"""

    print("  [Step 4 / Claude] Merge & 최종 확정...")
    return parse_json(query_claude(prompt))


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

    # ── 2. Config / file sanity ─────────────────────────
    for required in ["configs/default.yaml", "configs/fast.yaml", "scripts/smoke_test.py"]:
        if not (pkg_dir / required).exists():
            errors.append(f"missing required file: {required}")

    # ── 3. smoke test ───────────────────────────────────
    smoke_ok = False
    if syntax_ok:
        cmd = [sys.executable, "scripts/smoke_test.py", "--config", "configs/fast.yaml"]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180, cwd=str(pkg_dir)
            )
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
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, cwd=str(pkg_dir)
            )
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

    ok = syntax_ok and smoke_ok
    return {
        "ok":                  ok,
        "syntax_ok":           syntax_ok,
        "smoke_ok":            smoke_ok,
        "forward_ok":          forward_ok,
        "metric_contract_ok":  metric_contract_ok,
        "output_contract_ok":  output_contract_ok,
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
    proposals_dir.mkdir(exist_ok=True)
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
    errors  = validation_report.get("errors", [])
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
    result = parse_json(query_claude(prompt))

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

    # ── Step 3: Gemini 설계 리뷰 ─────────────────────────
    gemini_review = _gemini_design_review(base, spec, hypothesis)
    gem_path      = _save_proposal(pkg_dir, "gemini_review", gemini_review)

    # ── Step 4: Claude Merge ──────────────────────────────
    merged = _claude_merge(base, gpt_proposal, gemini_review, spec, hypothesis)
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
        blocked_reason = (
            "post-generation validation failed after repair"
            if final_val_report.get("repaired")
            else "post-generation validation failed"
        )
        blocked_msg = "Package generation blocked: final validation failed after one repair pass."
        print(f"\n  ❌ {blocked_msg}")
        print(f"     errors: {final_val_report.get('errors', [])}")
        print(f"     (artifacts retained for debugging: {pkg_dir})")
        # artifacts/proposals/validation reports는 디버깅을 위해 디스크에 유지
        # _finalize_package()는 호출하지 않음 — finalization marker 없음
        return {
            "pkg_dir":            str(pkg_dir),
            "spec_id":            spec["spec_id"],
            "hypothesis_id":      spec["hypothesis_id"],
            "experiment_version": version,
            "success":            False,
            "finalized":          False,
            "validation_failed":  True,
            "blocked_reason":     blocked_reason,
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
