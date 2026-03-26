# Pipeline 모듈 상세

## 아키텍처 흐름

```mermaid
flowchart LR
    IN([사용자 입력]) --> T

    subgraph Research["연구 단계"]
        T[1 topic_analyzer] --> P
        P[2 paper_researcher] --> H
        H[3 hypothesis_generator] --> V
        V[4 hypothesis_validator\nGPT-4o · Gemini] --> U
        U{5 user_approval\nPDF 보고서}
        U -->|거절/수정| H
    end

    subgraph Build["빌드 단계"]
        U -->|승인| CA
        CA[6 code_analyzer\nGitHub API] --> MG
        MG[7 model_generator\nClaude] --> RL
        RL[8 research_loop\nPyTorch]
        RL -->|목표 미달| MG
    end

    T -->|topic_analysis.json| R[("experiments/{slug}/reports/")]
    P -->|papers.json| R
    H -->|hypothesis.json| R
    CA -->|code_analysis.json| R
    MG -->|experiments/{slug}/runs/v{N}/| EX[("experiments/{slug}/runs/")]
    RL -->|result_summary.json\nprevious_results.jsonl| RE[("experiments/{slug}/results/")]
    RL -->|목표 달성| Done([완료])
```

## 모듈별 역할

| 모듈 | 역할 | 입력 | 출력 |
|---|---|---|---|
| topic_analyzer.py | 연구 주제 구조화 | 사용자 입력 4종 | `{slug}/reports/topic_analysis.json` |
| paper_researcher.py | arXiv/Semantic Scholar 논문 검색 | topic_analysis.json | `{slug}/reports/papers.json` |
| hypothesis_generator.py | Claude로 가설 생성 | topic + papers | `{slug}/reports/hypothesis.json` |
| hypothesis_validator.py | GPT-4o + Gemini 검증 | hypothesis | `{slug}/reports/validation.json` |
| user_approval.py | PDF 보고서 생성 + 승인 CLI | topic + hypothesis + papers | `{slug}/reports/approval.json`, `report.pdf` |
| code_analyzer.py | GitHub 코드 분석 | topic + hypothesis | `{slug}/reports/code_analysis.json` |
| model_generator.py | Fabric 실험 패키지 생성 | topic + hypothesis + code_analysis | `{slug}/runs/vN/` (Fabric 패키지) |
| research_loop.py | 실험 실행 + Path A/B/C revision 루프 | pkg_dir + topic + hypothesis + code_analysis | `{slug}/results/vN/result_summary.json`, `{slug}/results/previous_results.jsonl`, `{slug}/reports/revision_request_vN.json` |

> `{slug}` = `experiments/{topic_slug}`

## 실행 명령

```bash
# 단계별 독립 실행
python -m lab.topic_analyzer --topic "..." --details "..." \
    --problem "..." --outcome "..." --constraints "..." --metric "PSNR, SSIM"
# 결과: experiments/{slug}/reports/topic_analysis.json

python -m lab.paper_researcher \
    --topic-file experiments/{slug}/reports/topic_analysis.json
# 결과: experiments/{slug}/reports/papers.json

python -m lab.hypothesis_generator \
    --topic-file  experiments/{slug}/reports/topic_analysis.json \
    --papers-file experiments/{slug}/reports/papers.json
# 결과: experiments/{slug}/reports/hypothesis.json

python -m lab.hypothesis_validator \
    --hypothesis-file experiments/{slug}/reports/hypothesis.json \
    --topic-file      experiments/{slug}/reports/topic_analysis.json
# 결과: experiments/{slug}/reports/validation.json

python -m lab.user_approval \
    --topic-file      experiments/{slug}/reports/topic_analysis.json \
    --hypothesis-file experiments/{slug}/reports/hypothesis.json \
    --papers-file     experiments/{slug}/reports/papers.json
# 결과: experiments/{slug}/reports/approval.json, report.pdf

python -m lab.code_analyzer \
    --topic-file      experiments/{slug}/reports/topic_analysis.json \
    --hypothesis-file experiments/{slug}/reports/hypothesis.json
# 결과: experiments/{slug}/reports/code_analysis.json

python -m lab.model_generator \
    --topic-file      experiments/{slug}/reports/topic_analysis.json \
    --hypothesis-file experiments/{slug}/reports/hypothesis.json \
    --code-file       experiments/{slug}/reports/code_analysis.json \
    --version 1
# 결과: experiments/{slug}/runs/v1/ (Fabric 패키지 디렉토리)

python -m lab.research_loop \
    --pkg-dir         experiments/{slug}/runs/v1 \
    --topic-file      experiments/{slug}/reports/topic_analysis.json \
    --hypothesis-file experiments/{slug}/reports/hypothesis.json \
    --code-file       experiments/{slug}/reports/code_analysis.json \
    --max-rounds 3 \
    --runner-type local   # 또는 github

# Path A: 자동으로 runs/v2, v3 재생성 후 재실행
# Path B/C: experiments/{slug}/reports/revision_request_v{N}.json 생성 후 종료
# 결과 (canonical, research_loop.py만 생성):
#   experiments/{slug}/results/vN/result_summary.json
#   experiments/{slug}/results/vN/runner_metadata.json
#   experiments/{slug}/results/previous_results.jsonl
```

## Runner 추상화

### RunResult 계약

모든 runner는 동일한 `RunResult` dict를 반환한다:

```python
{
  "status":       "success | failed | smoke_failed | timeout | metrics_parse_error",
  "metrics":      {"psnr": 28.5, ...},   # METRICS:{} stdout 파싱 결과
  "stdout_lines": [...],                  # 전체 stdout
  "stderr_tail":  [...],                  # stderr 마지막 200줄
  "returncode":   int,                    # 0=success, 1=failure, -1=runner 오류
  "metadata": {
    "runner":         "local | github",
    "job_id":         str,        # GitHub: run_id,    local: ""
    "pipeline_id":    str,        # GitHub: run_number, local: ""
    "dispatch_id":    str,        # GitHubActionsRunner UUID, local: ""
    "duration_s":     float,
    "artifact_uri":   str,        # GitHub artifact API URL
    "job_url":        str,        # GitHub: html_url (브라우저 링크)
    "git_sha":        str,
    "git_branch":     str,
    "experiment_pkg": str,        # pkg_dir 절대 경로
    "started_at":     str,        # ISO8601
    "finished_at":    str,        # ISO8601
  }
}
```

### 실패 semantics

| status | 원인 |
|---|---|
| `smoke_failed` | smoke_test.py exit 1 |
| `failed` | train 실패 (workflow failure, dispatch 실패, artifact 실패 등) |
| `timeout` | poll deadline 초과 |
| `metrics_parse_error` | METRICS:{} 라인 없음 또는 파싱 오류 |
| `success` | 정상 완료 + metrics 파싱 성공 |

### canonical result_summary 생성 책임

- `result_summary.json`은 **`research_loop.py`만** 생성한다
- workflow는 `final_metrics.json` + `runner_metadata.json`만 artifact로 업로드한다
- runner는 두 파일을 다운로드하여 `RunResult`를 구성하고 research_loop에 반환한다
- research_loop은 `experiment_spec.json` + `RunResult`를 결합하여 canonical summary를 생성한다

### artifact 계약 (GitHubActionsRunner)

workflow `experiment-results` artifact에 포함되는 파일:

```
artifacts/metrics/final_metrics.json   # raw metric dict (METRICS:{} 파싱 결과)
runner_metadata.json                   # 실행 메타데이터 (dispatch_id 포함)
```

## LLM 설정 (config.py)

| 역할 | 모델 | API |
|---|---|---|
| 논문 분석 / 가설 생성 / 코드 생성 | `claude-opus-4-6` | Anthropic |
| 가설 검증 / Research Loop 분석 | `gpt-5.1` | OpenAI |
| 가설 검증 / Research Loop 진단 | `gemini-2.5-pro` | Google |

```python
CLAUDE_MODEL = "claude-opus-4-6"
OPENAI_MODEL = "gpt-5.1"
GEMINI_MODEL = "gemini-2.5-pro"
```

## 데이터 스펙

### 연구 주제 입력
```json
{
  "topic": "deep learning denoising",
  "details": "산업 데이터, 1번 scan한 이미지 denoising",
  "problem_definition": "단일 scan으로 얻은 산업용 이미지는 noise가 심하여 결함 검출 정확도가 낮음",
  "desired_outcome": "PSNR 30dB 이상, 실시간 처리 가능한 경량 모델",
  "constraints": "경량 모델, inference time 중요",
  "target_metric": "PSNR, SSIM"
}
```

### 실험 결과 (result_summary.json 핵심 필드)
```json
{
  "schema_version": "1.0",
  "run_id":         "{slug}_v{N}_{YYYYMMDD_HHMMSS}",
  "hypothesis_id":  "...",
  "status":         "success | failed | smoke_failed | timeout | metrics_parse_error",
  "primary_metric": {"name": "psnr", "value": 28.5, "target": 30.0, "met": false},
  "secondary_metrics": [{"name": "ssim", "value": 0.81, "unit": ""}],
  "training_stability": {"loss_converged": true, "nan_detected": false},
  "recommended_next_actions": [{"path": "A", "rationale": "...", "priority": "high"}]
}
```
전체 스키마: schemas/result_summary.json
