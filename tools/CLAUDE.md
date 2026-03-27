# Tools — Claude Agent SDK Tool Registry

## 역할
`lab/` 모듈 함수와 1:1 매핑되는 tool 스키마 정의 및 실행 라우터.

## 현재 오케스트레이션 방식
`main.py`는 현재 **Bash 기반 오케스트레이션**을 사용한다:
- Claude Agent SDK의 `Bash/Read/Write/Edit` 도구로 각 단계를 직접 실행
- `tools/registry.py`의 `execute_tool()`은 **현재 main.py에서 호출되지 않음**
- registry는 향후 SDK tool_use 방식 전환 시 또는 외부 오케스트레이터 연동 시 사용

## 파일 구조

| 파일 | 역할 |
|---|---|
| `registry.py` | tool 스키마 목록 (`TOOLS`) + `execute_tool()` 라우터 |
| `__init__.py` | 패키지 초기화 |

## Tool 목록

| tool name | 매핑 함수 | 설명 |
|---|---|---|
| `analyze_topic` | `lab.topic_analyzer.analyze_topic` | 연구 주제 구조화 |
| `search_papers` | `lab.paper_researcher.search_papers` | arXiv/Semantic Scholar 논문 검색 |
| `generate_hypothesis` | `lab.hypothesis_generator.generate_hypothesis` | 가설 생성 |
| `validate_hypothesis` | `lab.hypothesis_validator.validate_hypothesis` | OpenAI + Gemini 검증 |
| `request_approval` | `lab.user_approval.request_approval` | PDF 보고서 생성 + 사용자 승인 |
| `analyze_github_code` | `lab.code_analyzer.analyze_code` | GitHub 코드 분석 |
| `generate_model` | `lab.model_generator.generate_experiment_package` | Fabric 실험 패키지 생성 (validation gate 포함) |
| `run_experiment` | `lab.research_loop.run_research_loop` | 실험 실행 + revision 루프 (multi-model 분석 + postcheck) |

## 규칙
- 새 tool 추가 시 `TOOLS` 리스트에 스키마 추가 + `execute_tool()` 라우터에 분기 추가
- tool input/output은 JSON 직렬화 가능해야 함
- 각 tool은 독립 실행 가능 — 오케스트레이터 없이도 `lab/` 모듈을 직접 호출 가능
