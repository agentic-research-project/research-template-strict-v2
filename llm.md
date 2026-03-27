# LLM 운용 가이드 — 박사급 연구 자동화

이 문서는 파이프라인 전체의 LLM 호출 품질을 **박사 10년차 수준**으로 통제하기 위한 중앙 가이드다.
모든 LLM 프롬프트는 이 가이드의 원칙을 따른다.

---

## 1. 모델 할당 및 역할

| 역할 | 상수 (config.py) | 기본값 | 사용 단계 |
|------|------------------|--------|----------|
| 코드 작성 + 감사 + 최종 결정 | `CLAUDE_MODEL` | claude-opus-4-6 | 1, 2, 3, 6, 7, 8, Decisive, Reframe |
| 비판 + 패치 + 해석 | `OPENAI_MODEL` | gpt-5.1 | 3, 4, 7, 8, Swing, Reframe |
| 독립 진단 + 설계 리뷰 | `GEMINI_MODEL` | gemini-2.5-pro | 3, 4, 7, 8, Swing, Reframe |

### 모델 변경 규칙

모델명은 `config.py` 한 곳에서만 관리한다. 사용자는 자유롭게 변경 가능:

```python
# lab/config.py — 이 3줄만 수정하면 전체 파이프라인 반영
CLAUDE_MODEL = "claude-opus-4-6"     # 예: "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-5.1"            # 예: "gpt-4o", "o1", "o3"
GEMINI_MODEL = "gemini-2.5-pro"     # 예: "gemini-2.5-flash", "gemini-3.1-pro"
```

**변경 시 주의사항**:
- `OPENAI_MODEL`을 o1/o3 계열로 변경 시: `response_format=json_object` 미지원 가능 → 파싱 오류 주의
- `GEMINI_MODEL`을 flash 계열로 변경 시: 추론 깊이 저하 가능 → 독립 진단 품질 확인 필요
- `CLAUDE_MODEL`을 sonnet/haiku로 변경 시: 코드 생성 품질 저하 가능 → mechanism audit 통과율 확인
- 모델 변경 후 첫 실행에서 비용/품질 트레이드오프를 확인할 것
- `LLM_QUERY_TIMEOUT`은 thinking 모델 기준(900s). 빠른 모델 사용 시 줄여도 됨

### 역할별 최소 요구 능력

| 역할 | 최소 요구 | 이유 |
|------|----------|------|
| Claude (코드 작성) | 코드 생성 + JSON 구조화 | model.py/module.py/data.py 전체 작성 |
| GPT (비판 + 해석) | JSON 출력 강제 가능 | `response_format=json_object` 의존 |
| Gemini (독립 진단) | 긴 입력 처리 + 독립 추론 | 전체 experiment context 수용 필요 |

### 역할 분리 원칙

- **Claude**: 생성 + 집행. 코드를 작성하고, 합의 결과를 집행한다. 독단적 판단 금지.
- **GPT**: 비판 + 제안. 약점을 공격하고, 패치를 제안한다. 코드를 직접 작성하지 않는다.
- **Gemini**: 독립 검증 + 중재. GPT 결과를 보지 않고(blind) 독립 판단한다. 코드를 작성하지 않는다.

> **핵심**: 어떤 LLM도 단독으로 최종 결정을 내리지 않는다. 합의 → 집행 → 사후검증 3단계를 거친다.

---

## 2. 호출 설정

| 설정 | 값 | 근거 |
|------|-----|------|
| Temperature (GPT) | `0.3` | 구조화된 JSON 출력의 일관성 확보. 창의성보다 정확성 우선 |
| Temperature (Gemini) | 기본값 | 독립 진단에 약간의 다양성 허용 |
| Temperature (Claude) | SDK 기본값 | thinking 모드에서 자체 조절 |
| Timeout | `LLM_QUERY_TIMEOUT` (900s) | Thinking 모델(o1, gemini-2.5-pro) 대응 |
| Retry | `LLM_QUERY_MAX_RETRIES` (2회) | Exponential backoff (1s, 2s) |
| Response format (GPT) | `{"type": "json_object"}` | 구조화된 출력 강제 |
| Prompt hash | 모든 호출에 `prompt_hash` 기록 | 재현성 추적 |

---

## 3. 박사급 프롬프트 원칙

### 반드시 포함해야 하는 요소

| 요소 | 설명 | 예시 |
|------|------|------|
| **역할 정의** | 구체적 전문 분야 + 행동 제약 | "당신은 PyTorch Fabric 전문 엔지니어이며, 유일한 코드 작성자입니다" |
| **입력 계약** | 최우선 참조 데이터 명시 | "experiment_spec을 최우선 계약으로 따르세요" |
| **출력 계약** | JSON 스키마 + 필수 필드 | `{"implemented": bool, "evidence": [...]}` |
| **근거 요구** | 판단에 대한 증거 제시 의무 | "evidence_used에 paper_id를 반드시 포함" |
| **제약 명시** | 하지 말아야 할 것 | "코드를 작성하지 마라", "reject된 패치를 적용하지 마라" |
| **반증 기준** | 실패 조건 정의 | "mechanism이 forward()에 없으면 미구현으로 판정" |

### 금지 패턴 (Anti-patterns)

| 금지 | 이유 | 대안 |
|------|------|------|
| "유망하다", "흥미롭다" | 근거 없는 낙관 | 구체적 수치 + 비교 대상 명시 |
| "여러 논문을 볼 때" | 모호한 참조 | paper_id 명시 (evidence_used) |
| "가능성이 있다" | 반증 불가 | "X 조건에서 Y 결과가 나오면 유효" |
| 전체 파일 재작성 | 변경 추적 불가 | 최소 diff (변경 부분만) |
| 점수만 높이 주기 | 증거 없는 고점 | score cap 규칙 적용 (evidence 부족 → 6점 상한) |

---

## 4. 단계별 LLM 계약

### Stage 1: 주제 분석 (Claude)

```
역할: 딥러닝 연구 전문가
입력: 사용자 연구 주제 (topic, details, constraints, target_metric)
출력: topic_analysis.json
```

**품질 기준**:
- 검색 키워드는 **phrase 형태** (3+ 단어). 단독 generic 단어 금지
- `constraints_structured`: regex 기반 구조화 (param_budget, single_gpu 등)
- `retrieval_plan`: 9개 query family (task-core ~ deployment-constraint-seeking)
- 도메인 자동 추론: 10개 클러스터 기반 target_metric 매핑

### Stage 2: 논문 검색 (Claude)

```
역할: 딥러닝 연구 분석가
입력: topic_analysis.json + 검색된 논문 목록
출력: papers.json (evidence_role, claim_slots, support_strength)
```

**품질 기준**:
- **sections 우선**: 논문에 sections(intro/method/experiment/results/limitation) 있으면 abstract 대신 참조
- `support_strength` 판정: sections 기반 = direct, abstract만 = indirect 상한
- Evidence coverage: novelty/validity/feasibility 3그룹 모두 충족 필수
- 테이블 + 수치 결과 자동 추출 (quantitative_results)

### Stage 3: 가설 생성 (Claude → GPT → Gemini → Claude)

```
5라운드 토론:
  Round 0 (Gemini): Evidence Pack 구조화
  Round 1 (Claude): 가설 3개 제안
  Round 2 (GPT): 비판 + pairwise ranking
  Round 3 (Gemini): 중재 + 합성
  Round 4 (Claude): 최종 확정 + falsification criteria
```

**품질 기준**:
- Coverage gate: 3개 미만 논문 또는 2+ 그룹 미충족 → `insufficient_evidence` 반환
- GPT 비판: "fatal_flaws", "exaggerations", "unverifiable" 집중
- Pairwise ranking: GPT + Gemini 독립 순위 → 승수 합산
- 최종 가설: falsification_criteria 필수 (측정 가능 + 비자명 + 방향성)
- `insufficient_evidence` / `no_robust_hypothesis` 안전 밸브 작동

### Stage 4: 가설 검증 (GPT + Gemini)

```
역할: 비판적 연구 리뷰어
입력: hypothesis.json + validation_packet
출력: validation.json (score_breakdown, verdict)
```

**품질 기준**:
- **3중 통과 기준**: 평균 ≥ 8.5, 각 항목 ≥ 8.0, 평가자 차이 ≤ 1.5
- **Prior-art comparison 테이블**: 3-5개 경쟁 논문별 강/약/잔여위험 비교 필수
- **Score cap (프로그램적 강제)**:
  - evidence_used 미인용 → novelty ≤ 6
  - evidence_links 불일치 → novelty + validity ≤ 6
  - constraints 미언급 → feasibility ≤ 6
  - falsification_criteria 미충족 → validity + impact ≤ 7
- **Falsification 품질 검증**: 측정 가능성 + 비자명성 + 반증 방향성 3기준
- **자동 개선 루프**: 최대 3회, 이전 강점 보존 + 약점 개선

### Stage 6: 코드 분석 (Claude)

```
역할: 딥러닝 코드 분석 전문가
입력: topic + hypothesis + GitHub 검색 결과
출력: code_analysis.json (reusable_components, architecture_insights)
```

**품질 기준**:
- 재사용 가능 컴포넌트: type (model/loss/dataset/trainer/utils) 분류
- adaptation_needed 필드로 수정 필요 사항 명시
- 코드 snippet은 핵심 10-20줄

### Stage 7: 모델 생성 (Claude + GPT + Gemini)

```
Claude: 코드 작성 (유일한 작성자)
GPT: 패치 제안 (spec-linked, 최소 변경)
Gemini: 설계 리뷰 (7차원 분석) + 패치 투표
Claude: 패치 투표 + merge 집행
```

**품질 기준**:
- **experiment_spec 최우선 계약**: 모든 코드가 spec의 architecture/training/evaluation에 바인딩
- **hypothesis_contract 구현 강제**: mechanism, target_metric, constraints가 코드에 반영
- **2-of-3 Patch Ballot**: GPT(implicit accept) + Gemini 투표 + Claude 투표 → 합의
  - `comparability_risk=high` → 무조건 reject
  - reject된 패치 → merge에서 제외 (Claude가 몰래 적용 금지)
- **3중 감사**:
  - mechanism_audit (Claude + AST 교차 검증): forward() 참여 여부
  - metric_audit (코드 검사): required_keys 존재
  - constraints_audit (패턴 매칭): param_budget, single_gpu 등
- **Hard gate**: 감사 실패 → 패키지 차단, 1회 수리 시도

### Stage 8: 실험 루프 (GPT → Gemini → Claude)

```
GPT: 결과 심층 해석 (suggested_path + evidence)
Gemini: 독립 진단 (GPT 결과를 보지 않음 — blind)
합의 레이어: consensus_strength 기반 path 결정
GPT: Path A 시 패치 제안
Claude: override reviewer (escalation만 허용)
사후검증: 결정론적 guardrail
```

**품질 기준**:
- **결정론적 진단 prior**: LLM 판단 전에 rule-based 진단 제공 (stability, bottleneck, confidence)
- **Gemini 독립성**: GPT 결과를 프롬프트에 포함하지 않음 (anchoring bias 방지)
- **consensus_strength**: confidence 가중 합의 점수 = (gpt_conf + gem_conf) / 2 × (1 - path_distance)
- **Path escalation 규칙**:
  - A → B: consensus_strength ≥ 0.35, n_runs ≥ 2
  - B → C: consensus_strength ≥ 0.55, n_runs ≥ 3, evidence=high
  - A → C: 금지 (skip 불가)
  - Downgrade: 금지
- **Postcheck**: 결정론적 최종 검증. Claude override가 조건 미충족 시 자동 복원
- **인과 추론**: 단일 변경 → 클린 효과 수집, 묶음 → 차분 귀인 + 가산 검증

---

## 4b. 신규 모듈 LLM 계약

### Decisive Evidence Compressor (`evidence_compressor.py`)

```
규칙 기반: importance_score 계산 (evidence_role × support_strength × rank × slots)
LLM 협업: GPT semantic swing 분석 → Gemini 독립 검증 → 합의
산출물: decisive_evidence.json (support/contra/swing/decision_pressure)
```

**Swing 합의 규칙**:
- GPT + Gemini 모두 swing 판정 → strong consensus
- paper_id 일치 + if_confirmed_changes 일치 → strong
- paper_id 일치 + effect 불일치 → weak
- 한쪽만 swing → 제외 (규칙 기반 fallback만 유지)

### Scientific Betting Engine (`scientific_betting.py`)

```
입력: evidence_coverage + decisive_evidence + constraints_structured + falsification_criteria
출력: scientific_bet.json (bet_type, info_gain, downside, minimal_test)
```

**bet_type 결정 (deterministic)**:
- `exploit`: support 강함 + contra 약함 + downside 낮음
- `probe`: evidence 희소 + info_gain 높음
- `reject`: contra 강함 + downside 높음

**downside 계산 (Expected Loss)**:
- `E[Loss] = P(wrong) × Impact(wrong)`
- P(wrong) ∝ contra 강도 + uncovered 비율
- Impact(wrong) ∝ constraint 엄격도

### Problem Reframing Layer (`problem_reframing.py`)

```
3자 협업:
  GPT: 4번째 frame (competing_explanation_driven) 제안
  Gemini: 4개 frame 4차원 점수 (falsifiability/feasibility/novelty/productivity) + 추천
  Claude: structured decision (followed_gemini/override + rejected_frames + decision_basis)
산출물: problem_reframing.json (frames[4], recommended_working_frame)
```

**Frame 선택 기준**: falsifiability 높고 + feasibility acceptable + novelty 보존

### A-7: Senior Researcher Diagnostics (`research_loop.py`)

```
_triage_pivotal_evidence: 판세를 바꾸는 핵심 근거 최대 3개 추출
_frame_scientific_bet: 희소 증거 기반 bet_grade (A~D) + hedge
_detect_problem_reframe: 문제 재정의 신호 4가지 감지
```

**pivotal_evidence**: ablation + bottleneck + effect_size에서 영향력 순 상위 3개
**scientific_bet**: confidence × evidence → A(강한 베팅) ~ D(보류). 단일 실행 → C 상한
**problem_reframe**: 2+ 신호 → `reframe_detected=true` + 구체적 제안

---

## 5. 증거 인용 규칙

### Evidence Pack → LLM 프롬프트 체인

```
papers.json (paper_id, evidence_role, claim_slots)
     ↓
hypothesis.json (evidence_links: [{paper_id, supports: [...]}])
     ↓
validation.json (evidence_used: [paper_id], prior_art_comparison: [...])
     ↓
experiment_spec.json (hypothesis_contract: {mechanism, target_metric, constraints})
     ↓
result_summary.json (hypothesis_implementation: {mechanism_audit, metric_audit, constraints_audit})
```

### 인용 규칙

| 규칙 | 적용 단계 | 강제 방법 |
|------|----------|----------|
| paper_id는 evidence_pack의 유효 ID만 사용 | 3, 4 | `_audit_evidence_links` |
| evidence_used 빈 배열 → novelty ≤ 6 | 4 | `_enforce_caps` |
| mechanism 미인용 → novelty ≤ 6 | 4 | evidence coverage matrix |
| 가설의 모든 claim은 paper_id로 추적 가능해야 함 | 3 | `_lint_hypothesis` |
| prior_art_comparison에 최소 3개 경쟁 논문 | 4 | 프롬프트 필수 조건 |

---

## 6. 응답 포맷 계약

### 공통 규칙

- **JSON만 출력** (마크다운 펜스, 설명 텍스트 금지)
- `parse_json()`이 자동으로 마크다운 펜스 제거하지만, LLM이 순수 JSON을 출력하는 것이 원칙
- 필수 필드 누락 시 `setdefault()`로 기본값 보장

### 단계별 출력 스키마

| 단계 | 출력 파일 | 필수 최상위 키 |
|------|----------|---------------|
| 1 | topic_analysis.json | research_question, search_keywords, success_criteria, constraints, target_metrics |
| 1+ | problem_reframing.json | frames, recommended_working_frame, reason, claude_decision |
| 2 | papers.json | papers (배열), evidence_coverage, decisive_evidence, search_log |
| 2+ | decisive_evidence.json | support, contra, swing, decision_pressure |
| 3 | hypothesis.json | statement, key_innovation, expected_mechanism, falsification_criteria, evidence_links |
| 4 | validation.json | score, score_breakdown, verdict, evidence_used, prior_art_comparison |
| 4+ | scientific_bet.json | bet_type, bet_confidence, info_gain, downside, minimal_test |
| 6 | code_analysis.json | reusable_components, architecture_insights, recommended_baseline |
| 7 | experiment_spec.json | spec_id, model_architecture, training_config, evaluation_config, hypothesis_contract |
| 8 | result_summary.json | primary_metric, training_stability, bottleneck_candidates, confidence_model, effect_size, pivotal_evidence, scientific_bet, problem_reframe |

---

## 7. 진단 엔진 상수 (config.py)

LLM이 결과를 해석할 때 참조하는 결정론적 prior:

| 상수 | 값 | 용도 |
|------|-----|------|
| CONVERGENCE_REL_THRESHOLD | 0.01 | 수렴 판정 (< 1% 변화) |
| PLATEAU_WINDOW | 3 | 정체 판정 (3 에포크 연속) |
| OVERFIT_GAP_THRESHOLD | 0.15 | 과적합 판정 (15% gap) |
| ATTAINMENT_FAILURE_RATIO | 0.5 | 실행 실패 수준 (< 50% 달성) |
| PATH_C_MIN_RUNS | 3 | 가설 기각 최소 실행 횟수 |
| CONSENSUS_STRENGTH_PATH_B | 0.35 | Path B 최소 합의 강도 |
| CONSENSUS_STRENGTH_PATH_C | 0.55 | Path C 최소 합의 강도 |
| EFFECT_SIZE_SMALL/MEDIUM/LARGE | 0.01/0.05/0.15 | 효과 크기 분류 |

---

## 8. LLM 호출 총량 및 비용 인식

| 단계 | 호출 수 | 주요 모델 |
|------|:-------:|----------|
| Stage 1 | 2 | Claude |
| Reframing | 3 | GPT + Gemini + Claude |
| Stage 2 | 2 | Claude |
| Decisive Evidence | 2 | GPT + Gemini (swing 협업) |
| Stage 3 | 6 | Claude + GPT + Gemini |
| Stage 4 | 2-8 | GPT + Gemini (+ Claude 개선 루프) |
| Stage 6 | 2 | Claude |
| Stage 7 | 8 | Claude + GPT + Gemini |
| Stage 8 | 4/round | GPT + Gemini + Claude |
| **1회 전체** | **~35** | **Path A 반복 시 +4/round** |

### 비용 절감 원칙

- 캐시: 각 단계 결과를 `experiments/{slug}/reports/`에 저장. 재실행 시 캐시 우선
- 토큰 절약: Stage 4 개선 루프는 최대 3회. 통과 여부 무관 Stage 5로 진행
- Stage 3은 1회만 실행 (재수립 금지)
- 코드 컨텍스트 절단: model.py[:2500], module.py[:2000] (토큰 비용 제어)

---

## 9. 문체 기준 (Decision Memo)

PDF 보고서 및 LLM 생성 텍스트에 적용되는 문체:

### 허용

- "Novelty depends on whether the constrained regime is materially different from the closest prior art."
- "The mechanism is plausible, but current evidence is indirect."
- "v2 changes improved stability but did not resolve the primary metric gap."

### 금지

- "이 연구는 매우 혁신적이며 큰 파급력을 가질 것으로 기대된다."
- "전반적으로 유망하고 좋은 방향으로 보인다."
- "여러 논문을 볼 때 충분히 가능성이 있다고 판단된다."

### 원칙

1. 짧고 단정한 문장
2. 근거 없는 낙관 금지. "가능성이 있다" → 왜 그런지 근거 첨부
3. "새롭다" → 무엇과 비교했는지 명시
4. "어렵다/리스크" → 어디서 깨질지 구체적으로
5. 내부 연구 검토 메모 톤 (논문 초록 X, 블로그 X)
