# experiments/claude.md — Experiment Code Generation & Revision Spec
# version: 1.0  updated: 2026-03-24

이 파일은 모든 실험 코드 생성·수정 시 Claude가 준수해야 하는 규칙 명세다.
코드 생성 전 반드시 이 파일을 읽고 모든 섹션을 준수한다.

---

## §1 SOURCE PRIORITY (불변)

코드 생성 시 아래 우선순위를 엄격히 따른다:

```
hypothesis.json          ← 과학적 진실 (가장 높음)
  ↓
experiment_plan.json     ← 실험 설계
  ↓
code_analysis.json       ← 참고용 구현 힌트 (직접 복사 금지)
  ↓
previous_results.jsonl   ← 개선 참고 (덮어쓰기 불가)
  ↓
experiments/CLAUDE.md    ← 현재 파일 (코딩 규칙)
```

- `hypothesis.json`의 mechanism, target_metric, constraints를 코드가 명시적으로 구현해야 한다.
- `code_analysis.json`은 참고용이며 직접 복사·붙여넣기 금지.
- 이전 결과(`previous_results.jsonl`)는 개선 방향을 알려주지만 실험 설계를 덮어쓰지 않는다.

---

## §2 FILE LAYOUT RULES

```
experiments/{topic_slug}/runs/v{N}/
├── train.py           # Claude 소유 — GPT/Gemini 직접 수정 불가
├── module.py          # Claude 소유 — GPT/Gemini 직접 수정 불가
├── model.py           # Claude 소유, GPT patch 허용 (proposals/ 경유)
├── data.py            # Claude 소유, GPT patch 허용 (proposals/ 경유)
├── configs/
│   ├── default.yaml   # Claude 소유 — 하이퍼파라미터 정의
│   ├── fast.yaml      # smoke test용 (2 epoch, 소량 데이터)
│   └── ablation/      # 각 ablation variant별 yaml
├── scripts/
│   ├── smoke_test.py  # forward pass + 2 step, exit 0/1
│   └── eval.py        # 저장된 체크포인트 평가
├── tests/
│   └── test_forward.py  # shape/dtype 단위 테스트 (GPT 제안 가능)
├── artifacts/
│   ├── checkpoints/   # best.ckpt, last.ckpt
│   ├── logs/          # metrics.csv, tensorboard
│   └── metrics/       # per_epoch_metrics.jsonl
├── README.md          # 자동 생성 실행 요약
└── claude.md          # 이 파일의 복사본 (버전 고정)
```

**소유권 규칙:**
- Claude writes: train.py, module.py, model.py, data.py, configs/, scripts/, tests/, README.md
- GPT may propose patches (Path A 한정): model.py, module.py, configs/default.yaml
- GPT must NOT write: train.py, data.py, hypothesis.json, experiment_plan.json
- Gemini (model_generator): 설계 리뷰만 → proposals/gemini_review_{timestamp}.json
- Gemini (research_loop): 2차 진단만 → proposals/gemini_diagnosis_{timestamp}.json
- 모든 제안은 proposals/ 에 저장 후 Claude가 검토·병합

---

## §3 FABRIC CODING RULES

### 초기화
```python
# train.py에서만 Fabric을 초기화한다
fabric = Fabric(
    accelerator="auto",
    precision=config.get("precision", "bf16-mixed"),
    devices=config.get("devices", 1),
    strategy=config.get("strategy", "auto"),
)
fabric.launch()
```

- Fabric 인스턴스는 train.py 메인 블록에서 단 한 번 생성
- module.py, model.py에서 Fabric import/초기화 금지

### Wrap 순서 (반드시 이 순서)
```python
fabric.seed_everything(config["seed"])          # 1. 시딩 (데이터/모델 전)
model     = ModelClass(config)                   # 2. 모델 생성
optimizer = torch.optim.Adam(model.parameters()) # 3. 옵티마이저 생성
model, optimizer = fabric.setup(model, optimizer) # 4. Fabric wrap
train_loader = fabric.setup_dataloaders(train_loader) # 5. DataLoader wrap
```

### 재현성 규칙
- `fabric.seed_everything(config["seed"])` 를 DataLoader 생성 이전에 호출
- `num_workers` ≥ 1 이면 `worker_init_fn` 으로 각 worker seed 설정
- `torch.backends.cudnn.deterministic = True` (성능 저하 허용)

### Precision 기본값
- CUDA: `bf16-mixed`
- CPU fallback: `32-true`
- 설정: `configs/default.yaml`의 `precision` 키로 override 가능

### 체크포인트 정책
```python
# 저장 (best / last 두 종류만)
fabric.save(
    "artifacts/checkpoints/best.ckpt",
    {"model": model, "optimizer": optimizer, "epoch": epoch,
     "config": config, "best_metric": best_val}
)
# 로드
state = fabric.load("artifacts/checkpoints/best.ckpt")
model.load_state_dict(state["model"])
```
- 체크포인트에 항상 config 포함 (재현성)
- epoch마다 last.ckpt 덮어씀; val metric 개선 시에만 best.ckpt 업데이트

---

## §4 OUTPUT CONTRACT RULES

### METRICS stdout 계약 (불변)
- train.py는 학습 완료 후 마지막 줄로 정확히 아래 형식을 출력한다:
```
METRICS:{"psnr": 28.5, "ssim": 0.82, "params_M": 3.1, "inference_ms": 12.4}
```
- 패턴: `^METRICS:\{.*\}$` (한 줄, 공백 없음)
- 값은 반드시 float (정수도 float으로 캐스팅)
- 이 줄이 없거나 파싱 실패 시 GitLab CI는 `metrics_parse_error`로 처리

### Epoch 로깅 형식
```python
# artifacts/logs/metrics.csv 에 매 epoch 기록
# artifacts/metrics/per_epoch_metrics.jsonl 에 JSON Lines 형식으로 기록
{"epoch": 1, "train_loss": 0.42, "val_psnr": 27.1, "val_ssim": 0.79, "lr": 1e-4}
```

---

## §5 REVISION RULES

| Path | 조건 | hypothesis.json | experiment_plan.json |
|---|---|---|---|
| **A** — 실험/코드 수정 | 구현 문제, 불안정 학습, 약한 baseline | 변경 금지 | 수정 가능 |
| **B** — 가설 정제 | 핵심 아이디어 유효하나 범위 과대 | 좁히기만 가능 (refined_from 추가) | 재생성 |
| **C** — 가설 교체 | ≥3회 실행 + 강한 반증 | 교체 (supersedes 링크 유지) | 재생성 |

**에스컬레이션 원칙:**
- Path A → B → C 순서로만 에스컬레이션
- 단 1회 실패는 Path C 트리거 불가
- Path C는 Claude가 명시적 falsification 근거를 작성한 후에만 실행

**Path B 명시적 허용 조건 (모두 충족 필요):**
- GPT `suggested_path` ∈ {B, C}
- Gemini `agreement_with_gpt` ≠ "disagree"
- Consensus `agreement_level` ∈ {medium, strong}
- `n_runs ≥ 2` (단일 smoke/execution 실패만으로는 B 불가)

**Path C 명시적 허용 조건 (모두 충족 필요):**
- `n_runs ≥ 3`
- Consensus `agreement_level == "strong"`
- GPT `evidence_strength == "high"`
- `escalation_risk ≠ "blocked"`
- 훈련 불안정(NaN, 미수렴)이 아닌 가설 핵심 메커니즘 실패일 것

**Post-check guardrail:** Claude 결정 직후 `_postcheck_final_decision()`이 위 조건을 코드 레벨에서 재검증. 미충족 시 자동 다운그레이드.

---

## §6 ANTI-DRIFT RULES

- 선언되지 않은 외부 의존성 추가 금지 (`requirements.txt` 변경 시 별도 승인)
- hypothesis.json에 없는 기능(feature) 추가 금지
- 실험 범위 밖 최적화(ex: 다른 태스크에 유용한 모듈 추가) 금지
- `TODO`, `FIXME`, `HACK` 주석은 revision_request.json에 기록 후 제거
- 코드 주석: 로직이 자명하지 않은 경우에만 추가

---

## §7 PUBLIC-CODE REUSE RULES

- 공개 레포지토리 코드 연속 20줄 이상 직접 복사 금지
- 참고한 경우 파일 상단 주석에 출처 명시:
  ```python
  # Adapted from: https://github.com/xxx/yyy (license: MIT)
  ```
- code_analysis.json의 `reusable_components`는 구조 참고용; 직접 복사 금지
- GPT 패치 병합 시 Claude가 20줄 규칙 위반 여부 확인 (merge checklist §7)

---

## §8 COMPARABILITY RULES

- metric 이름은 v1에서 확정 후 동일 가설 cycle 전체에서 변경 불가
- configs/default.yaml은 이전 버전 상속 (`inherits: ../산업용_..._v{N-1}/configs/default.yaml`)
- 동일 가설 하의 모든 run은 동일 test set으로 평가
- ablation run은 반드시 동일 패키지 버전 내에서 실행 (다른 버전과 비교 금지)
- baseline_run_id는 result_summary.json에 항상 기록

---

## §9 METRIC NAMING RULES (정식 명칭)

| Metric | 단위 | 키 이름 |
|---|---|---|
| Peak Signal-to-Noise Ratio | dB | `psnr` |
| Structural Similarity Index | — (0~1) | `ssim` |
| Model parameter count | 백만 | `params_M` |
| Inference time (single image) | ms | `inference_ms` |
| Training loss (final epoch) | — | `train_loss` |
| Validation loss (best epoch) | — | `val_loss` |
| Defect detection F1 | — (0~1) | `defect_f1` |
| FLOPs (per forward pass) | GFLOPs | `gflops` |

- 새 metric 추가 시 이 표에 등록 후 사용
- 약어, 대문자, 단위 혼입 금지 (`PSNR_db`, `psnr_db` 사용 불가 → `psnr` 만 허용)

---

## §10 PATCH/UPDATE DISCIPLINE

### GPT 패치 형식
```json
{
  "target_file": "model.py|module.py|configs/default.yaml",
  "spec_field": "model_architecture|training_config|evaluation_config",
  "rationale": "conv3x3 → depthwise separable conv로 교체해 params 감소",
  "hypothesis_alignment_check": "hypothesis.json §constraints: params ≤ 5M에 부합",
  "breaks_comparability": false,
  "complexity_delta_loc": 12,
  "changes": [
    {"type": "replace", "old": "exact snippet", "new": "improved snippet"}
  ]
}
```
저장 위치: `proposals/gpt_patch_{YYYYMMDD_HHMMSS}.json`

- `spec_field`: 패치가 연결된 experiment_spec 섹션 (필수)
- `breaks_comparability`: 이전 run과의 비교 가능성 파괴 여부 (필수, true이면 Claude 명시 승인 필요)
- full-file rewrite 금지 — 최소 diff만 허용
- Path A 결정 시에만 패치 생성 (Path B/C/done에서는 패치 생성하지 않음)

### Claude 병합 체크리스트
병합 전 반드시 `docs/merge_checklist.md`의 항목을 모두 확인한다.

### 버전 업 조건
새 `runs/v{N}/` 패키지를 생성하는 경우:
- Path A: config 또는 코드 변경이 실질적일 때 (minor tweak은 동일 버전 허용)
- Path B/C: 항상 새 버전 생성

---

## §11 VALIDATION GATE (패키지 생성 차단 정책)

패키지 생성 완료 후 `_validate_generated_package()`가 아래 항목을 검사한다:
- 필수 파일 존재 여부 (train.py, model.py, module.py, data.py, configs/, scripts/)
- `METRICS:` stdout 패턴 존재 여부 (output_contract)
- metric 키 계약 준수 여부 (experiment_spec의 `required_keys`)

**검증 실패 시:**
- `finalized=False`, `validation_failed=True` 반환
- `_finalize_package()` 호출 차단
- 생성된 파일은 디버깅용으로 보존
- CLI는 `sys.exit(1)` 종료
- research_loop에서는 루프 중단 + revision_request 생성

---

## §12 MULTI-MODEL ANALYSIS PIPELINE (research_loop)

실험 실행 후 결과 분석은 5단계 파이프라인으로 처리된다:

```
1. GPT(interpret)  → 결과 심층 해석 + suggested_path (주요 해석자)
2. Gemini(diagnose) → 독립 2차 진단, agreement_with_gpt (짧은 의견)
3. consensus        → GPT+Gemini 통합, Path C 과도 에스컬레이션 방지 (코드 로직)
4. GPT(patches)     → Path A 한정, 구현 패치 제안 (consensus==A 시에만)
5. Claude(decide)   → 최종 결정 (GPT 해석 + Gemini 진단 + consensus 통합)
   + postcheck      → Path B/C guardrail 코드 레벨 재검증 (LLM 없음)
```

**파이프라인 불변 규칙:**
- GPT 해석 단계에서 코드 작성 금지
- Gemini 진단 단계에서 코드 작성 금지
- GPT 패치는 Path A 결정 시에만 생성 (B/C/done에서는 생략)
- Claude는 독자적 재해석 대신 제공된 증거를 통합하여 결정
- postcheck이 조건 미충족 결정을 자동 다운그레이드 (B→A, C→B 또는 A)

---

## §13 EXPERIMENT PACKAGE TEMPLATE

새 패키지 생성 시 `experiments/template/` 을 복사하여 시작한다:
```bash
cp -r experiments/template experiments/{topic_slug}/runs/v{N}
```
템플릿 구조는 `experiments/template/` 참조.
생성 후 §11 Validation Gate 통과 필수.
