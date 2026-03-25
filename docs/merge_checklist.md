# Merge Checklist — Claude / GPT-Codex / Gemini Proposals
# version: 1.0  (experiments/CLAUDE.md §10 참조)

Claude는 모든 제안(proposals/)을 병합하기 전 이 체크리스트를 완료해야 한다.
모든 항목이 ✅이어야 병합 가능. ❌가 하나라도 있으면 제안 거절 또는 수정 요청.

---

## §1 공통 사전 확인 (모든 제안)

- [ ] 제안 파일 형식이 올바른가?
  - GPT: `proposals/gpt_patch_{timestamp}.diff` (JSON 포함 unified diff)
  - Gemini: `proposals/gemini_review_{timestamp}.json`
- [ ] 제안의 `hypothesis_alignment_check` 또는 `verdict` 필드가 존재하는가?
- [ ] 제안이 `train.py` 또는 `module.py`를 수정하려 하는가?
  - ✅ 허용: model.py, data.py, tests/, configs/ablation/
  - ❌ 불허: train.py, module.py, hypothesis.json, experiment_plan.json

---

## §2 GPT Patch 검토

- [ ] **spec 연결**: patch에 `spec_field` 필드가 존재하는가?
  (`model_architecture | training_config | evaluation_config` 중 하나)
- [ ] **비교 가능성**: `breaks_comparability` 필드가 명시됐는가?
  - `true`인 경우 Claude가 명시적으로 승인해야 함
- [ ] **가설 정합성**: patch의 `hypothesis_alignment_check`가 hypothesis.json의 mechanism/constraints와 일치하는가?
- [ ] **범위**: full-file rewrite가 아닌 최소 diff인가? (전체 파일 재작성 금지)
- [ ] **복잡도**: `complexity_delta_loc` ≤ 50 LOC 인가?
- [ ] **적용 가능성**: patch가 현재 target_file에 깨끗이 적용되는가?
- [ ] **smoke test**: patch 적용 후 `python scripts/smoke_test.py` exit 0인가?
- [ ] **단위 테스트**: `python -m pytest tests/` 통과하는가?
- [ ] **공개 코드 규칙**: 연속 20줄 이상의 공개 코드 복사가 없는가?
  - 있다면 상단 주석에 출처(URL + 라이선스) 명시됐는가?
- [ ] **새 의존성**: 새 외부 패키지가 없는가? 있다면 requirements.txt에 추가되었는가?
- [ ] **매직 넘버**: 코드 중간에 설명 없는 상수가 없는가? (config에서 읽어야 함)

---

## §3 Gemini 리뷰 검토

Gemini는 **두 가지 역할**로 분리된다.

### §3-A. model_generator — 실험 설계 리뷰 (`gemini_review_*.json`)

- [ ] `verdict`가 허용 값 중 하나인가?
  (`accept_as_is | accept_with_patch | revise_experiment | refine_hypothesis | replace_hypothesis`)
- [ ] `mechanism_check`, `spec_alignment`, `bottlenecks` 필드가 존재하는가?
- [ ] `evidence_strength`가 명시됐는가? (low/medium/high + 근거)
- [ ] `suggested_path`가 evidence_strength와 일치하는가?
  - low → Path A만 허용
  - medium → Path A 또는 B
  - high → Path C 허용 (단, 3+ 실패 run 필요)
- [ ] Gemini가 직접 코드를 작성했는가? → ❌ 거절 (설계 리뷰만 허용)

### §3-B. research_loop — 독립 2차 진단 (`gemini_diagnosis_*.json`)

- [ ] `suggested_path` 필드가 존재하는가? (A|B|C|done)
- [ ] `short_diagnosis` 필드가 존재하는가? (한 문장 진단)
- [ ] `agreement_with_gpt` 필드가 존재하는가? (agree|partial|disagree)
- [ ] `disagreement_reason` 필드가 partial/disagree 시 명시됐는가?
- [ ] `main_risk` 필드가 존재하는가?
- [ ] `confidence` 필드가 존재하는가? (low|medium|high)
- [ ] Gemini가 코드를 작성하거나 패치를 제안했는가? → ❌ 거절 (진단만 허용)

---

## §4 비교 가능성 확인

- [ ] metric 키 이름이 변경됐는가? → ❌ 불허 (§8 §9 고정 규칙)
- [ ] 새 버전의 `configs/default.yaml`이 이전 버전의 필수 키를 모두 포함하는가?
- [ ] ablation run이 같은 `_v{N}` 패키지 내에서 실행되는가?
- [ ] `deltas_vs_baseline`의 `baseline_run_id`가 이전 run을 올바르게 가리키는가?

---

## §5 Path A/B/C 결정 전 최종 확인

### 5-A. Path B 허용 조건 (모두 충족 필요)
- [ ] GPT `suggested_path` ∈ {B, C}인가?
- [ ] Gemini `agreement_with_gpt` ≠ "disagree"인가?
- [ ] Consensus `agreement_level` ∈ {medium, strong}인가?
- [ ] `n_runs ≥ 2`인가?
- [ ] 단순 smoke/execution 실패가 아닌가? (smoke_failed 단독으로는 B 불가)

### 5-B. Path C 허용 조건 (모두 충족 필요)
- [ ] `n_runs ≥ 3`인가?
- [ ] Consensus `agreement_level == "strong"`인가?
- [ ] GPT `evidence_strength == "high"`인가?
- [ ] `escalation_risk ≠ "blocked"`인가?
- [ ] 훈련 불안정(NaN, 미수렴)이 아닌 가설 핵심 실패인가?
- [ ] Path C 결정 전 Path A와 Path B가 모두 불가능함을 확인했는가?
- [ ] `supporting_run_ids`에 ≥3개의 독립 실패 run이 포함됐는가?

### 5-C. 공통 확인
- [ ] `_postcheck_final_decision()` 결과가 원래 결정과 일치하는가?
  - 불일치 시 `postcheck_override=True`, `postcheck_note` 확인 필요
- [ ] Path B/C의 경우 새 experiment_plan.json이 업데이트된 hypothesis.json을 기반으로 재생성됐는가?
- [ ] revision_request.json의 `justification` 필드가 ≥50자의 구체적 근거를 포함하는가?

---

## §6 병합 완료 후 기록

병합 후 아래를 수행한다:
1. `proposals/` 내 처리된 파일에 `_accepted` 또는 `_rejected` suffix 추가
2. `revision_request.json` 저장 (Path A/B/C 해당 시)
3. `experiments/{slug}_v{N}/claude.md` 이 파일의 현재 버전으로 업데이트
4. `previous_results.jsonl` 에 run 결과 append 확인
