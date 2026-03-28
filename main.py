"""
AI-Driven Deep Learning Research Automation
메인 오케스트레이터 (Claude Agent SDK)

파이프라인:
  1단계  주제 분석       topic_analyzer
  2단계  논문 검색       paper_researcher  (최신 2년 → 부족 시 1년씩 소급)
  3단계  협업 가설 수립  hypothesis_generator --mode collaborative
           Claude(제안) → OpenAI(비판+대안) → Gemini(중재+합성) → Claude(최종)
  4단계  검증+자동개선   hypothesis_validator --refine --target-score 9.0
           9점 미달 → Claude가 약점 개선 후 재검증 (최대 3회)
           9점 미달 지속 → 3단계로 복귀
  5단계  사용자 승인     user_approval  (PDF 보고서 생성 포함)
  6단계  코드 분석       code_analyzer
  7단계  모델 생성       model_generator
  8단계  실험 실행       research_loop

실행:
  python main.py                              # 기본 예시 주제로 실행
  python main.py --topic "..." --details "..." ...   # 직접 주제 입력
"""

import anyio
import argparse
import json
import re
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, AssistantMessage
from lab.config import query_claude, parse_json, validate_stage_preconditions, SCORE_THRESHOLD as PDF_SCORE_THRESHOLD


# ──────────────────────────────────────────────────────────
# 시스템 프롬프트
# ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 딥러닝 연구를 자동화하는 AI 연구자입니다.
주어진 연구 주제에 대해 아래 8단계를 순서대로 진행하세요.

각 단계 시작 시 반드시 "[단계 N 시작]" 을 출력하고, 완료 시 "[단계 N 완료]" 를 출력하세요.

────────────────────────────────────────────────────────
## 워크스페이스 구조
────────────────────────────────────────────────────────
모든 파일은 주제별 workspace에 저장된다:
  experiments/{topic_slug}/
    reports/         ← 1~6단계 JSON 보고서
    results/         ← 실험 결과 (vN/result_summary.json, previous_results.jsonl)
    runs/vN/         ← 실험 패키지 (train.py, model.py, configs/ 등)

────────────────────────────────────────────────────────
## 1단계: 주제 분석
────────────────────────────────────────────────────────
Bash로 아래 명령을 실행한다:
  python -m lab.topic_analyzer

실행이 불가하면 직접 분석하여 아래 형식으로
experiments/{topic_slug}/reports/topic_analysis.json 을 Write로 저장한다:
{
  "input": {topic, details, problem_definition, desired_outcome, constraints, target_metric},
  "research_question": "핵심 연구 질문 (영어)",
  "search_keywords": {"primary": [...], "secondary": [...]},
  "success_criteria": {"quantitative": [...], "qualitative": [...]},
  "constraints": [...],
  "target_metrics": [...]
}

────────────────────────────────────────────────────────
## 2단계: 논문 검색  (최신 2년 우선, 부족 시 1년씩 소급)
────────────────────────────────────────────────────────
Bash로 실행:
  python -m lab.paper_researcher \
    --topic-file experiments/{topic_slug}/reports/topic_analysis.json

실행 후 결과 파일을 Read로 확인하고 논문이 10편 미만이면 재실행한다.
결과: experiments/{topic_slug}/reports/papers.json

────────────────────────────────────────────────────────
## 3단계: 협업 가설 수립  ← 핵심 단계
────────────────────────────────────────────────────────
Bash로 실행 (반드시 --mode collaborative 사용):
  python -m lab.hypothesis_generator \
    --topic-file  experiments/{topic_slug}/reports/topic_analysis.json \
    --papers-file experiments/{topic_slug}/reports/papers.json \
    --mode collaborative

5라운드 토론이 진행된다:
  Round 0 — Gemini:  Evidence Pack 구조화 (논문 → 문제/방법/한계/갭)
  Round 1 — Claude:  Evidence Pack 기반 가설 3개 제안
  Round 2 — GPT:     3개 가설 공격적 비판 + 최선 선택
  Round 3 — Gemini:  중재 + 합성 (1~2개 정제 가설)
  Round 4 — Claude:  최종 가설 + falsification_criteria

⚠️ 만약 hypothesis.json에 "status": "insufficient_evidence" 또는
"no_robust_hypothesis"가 있으면 4-5단계를 건너뛰고 사용자에게 보고한다.
사용자가 추가 키워드를 제공하면 2단계부터 재시작할 수 있다.

완료 후 debate_summary를 AskUserQuestion으로 사용자에게 보고한다:
  "3단계 완료 — 협업 가설 수립 결과:
   - Claude 제안 3개: ...
   - GPT 비판/최선: ...
   - Gemini 합성: ...
   - 최종 가설 (KR): ..."
결과: experiments/{topic_slug}/reports/hypothesis.json

────────────────────────────────────────────────────────
## 4단계: LLM 검증 + 자동 개선 루프  (목표: 8.5점)
────────────────────────────────────────────────────────
Bash로 실행:
  python -m lab.hypothesis_validator \
    --hypothesis-file experiments/{topic_slug}/reports/hypothesis.json \
    --topic-file      experiments/{topic_slug}/reports/topic_analysis.json \
    --refine --target-score 8.5 --max-iter 3

결과를 Read로 확인한다.

통과 기준 (3가지 모두 충족):
  - 평균 점수 ≥ 8.5
  - 각 항목(novelty/validity/feasibility/impact) 평균 ≥ 8.0
  - 두 평가자 점수 차이 ≤ 1.5

  - 루프 내에서 최대 3회 Claude가 약점을 개선하며 재검증한다
  - 통과 여부와 무관하게 항상 5단계로 진행한다 (토큰 낭비 방지)
  - 최고 점수를 기록한 가설이 validation.json ["hypothesis"] 필드에 저장된다
  - 3단계(협업 생성)로 절대 돌아가지 않는다

결과: experiments/{topic_slug}/reports/validation.json

────────────────────────────────────────────────────────
## 5단계: 사용자 승인  (PDF 보고서 — 점수 무관 항상 생성)
────────────────────────────────────────────────────────
Bash로 실행:
  python -m lab.user_approval \
    --topic-file      experiments/{topic_slug}/reports/topic_analysis.json \
    --hypothesis-file experiments/{topic_slug}/reports/hypothesis.json \
    --validation-file experiments/{topic_slug}/reports/validation.json \
    --papers-file     experiments/{topic_slug}/reports/papers.json \
    [--auto-approve]

  → 통과: 정상 PDF (초록 푸터)
  → 미달: 경고 배너 PDF (주황 푸터) + 최고 점수 가설로 보고
  → --auto-approve 시: input() 건너뛰고 자동 approve

사용자 결정에 따라 (대화 모드):
  approve → 6단계 진행 (미달이어도 사용자가 승인 가능)
  revise  → 사용자 수정 의견 기록 후 파이프라인 종료 (수동 재시작)
  reject  → 파이프라인 종료

자동 승인 모드 (--auto-approve):
  PDF 생성 후 즉시 approve 처리, 6단계로 진행

결과: experiments/{topic_slug}/reports/approval.json
      experiments/{topic_slug}/reports/report.pdf

────────────────────────────────────────────────────────
## 6단계: GitHub 코드 분석
────────────────────────────────────────────────────────
Bash로 실행:
  python -m lab.code_analyzer \
    --topic-file      experiments/{topic_slug}/reports/topic_analysis.json \
    --hypothesis-file experiments/{topic_slug}/reports/hypothesis.json

결과: experiments/{topic_slug}/reports/code_analysis.json

────────────────────────────────────────────────────────
## 7단계: PyTorch Fabric 실험 패키지 생성
────────────────────────────────────────────────────────
Bash로 실행:
  python -m lab.model_generator \
    --topic-file      experiments/{topic_slug}/reports/topic_analysis.json \
    --hypothesis-file experiments/{topic_slug}/reports/hypothesis.json \
    --code-file       experiments/{topic_slug}/reports/code_analysis.json \
    --version 1

이 명령은 experiments/template/ 을 복사한 후
Claude가 model.py / module.py / data.py / configs/default.yaml을 가설 기반으로 생성한다.

결과: experiments/{topic_slug}/runs/v1/   (패키지 디렉토리)
      experiments/{topic_slug}/runs/v1/experiment_spec.json

────────────────────────────────────────────────────────
## 8단계: 실험 실행 + Revision 루프
────────────────────────────────────────────────────────
Bash로 실행:
  python -m lab.research_loop \
    --pkg-dir         experiments/{topic_slug}/runs/v1 \
    --topic-file      experiments/{topic_slug}/reports/topic_analysis.json \
    --hypothesis-file experiments/{topic_slug}/reports/hypothesis.json \
    --code-file       experiments/{topic_slug}/reports/code_analysis.json \
    --max-rounds 3 \
    --runner-type {{runner_type}}

루프 동작:
  - 각 round: smoke test → train → METRICS 파싱 → result_summary.json 생성
  - 목표 달성: 종료
  - Path A (코드 수정): 자동으로 runs/v{N+1} 패키지 재생성 후 재실행 (최대 3회)
  - Path B/C: revision_request.json 생성 후 종료

실험 완료 후 result_summary.json 을 Read로 읽어 분석하고 최종 보고서를 작성한다.
결과: experiments/{topic_slug}/results/vN/result_summary.json
      experiments/{topic_slug}/results/previous_results.jsonl  (모든 run append)
      experiments/{topic_slug}/reports/revision_request_v{N}.json  (Path B/C 시)

────────────────────────────────────────────────────────
## 전체 규칙
────────────────────────────────────────────────────────
- topic_slug = 연구 주제를 소문자로 변환 후 공백/특수문자를 _로 치환한 앞 30자
  (예: "deep learning denoising" → "deep_learning_denoising")
- 각 단계 결과는 반드시 지정된 experiments/{topic_slug}/ workspace에 저장한다
- Bash 명령 실패 시 에러를 Read로 확인하고 원인을 파악한 뒤 재시도한다
- 사용자 승인(5단계) 없이 6단계로 진행하지 않는다
- 3단계(협업 가설)는 1회만 실행한다 — 토큰 절약을 위해 재수립하지 않는다
- 개선은 4단계 refinement 루프(최대 3회) 내에서만 수행한다
- 5단계 PDF는 점수 무관 항상 생성한다
- 7단계 결과는 단일 .py 파일이 아닌 패키지 디렉토리(experiments/{slug}/runs/vN/)다
- 8단계 --pkg-dir 인자에는 패키지 디렉토리 경로를 지정한다

────────────────────────────────────────────────────────
## ⚠️ 단계 선행 조건 검증 (Precondition Guard)
────────────────────────────────────────────────────────
각 단계를 시작하기 전에 아래 파일이 존재하는지 반드시 Read로 확인하라.
파일이 없으면 해당 단계를 실행하지 말고 "[오류] N단계 선행 조건 미충족: {파일명}" 을 출력하라.

| 단계 | 필수 선행 파일 |
|------|---------------|
| 2 | topic_analysis.json |
| 3 | topic_analysis.json, papers.json |
| 4 | hypothesis.json |
| 5 | hypothesis.json, validation.json |
| 6 | hypothesis.json, approval.json (decision=approve) |
| 7 | hypothesis.json, code_analysis.json |
| 8 | runs/v1/experiment_spec.json |

이 검증을 건너뛰면 파이프라인이 불완전한 데이터로 실행되어 결과가 무효화된다.
"""


# ──────────────────────────────────────────────────────────
# todo.md 파서
# ──────────────────────────────────────────────────────────

def parse_todo(todo_path: str) -> dict:
    """todo.md를 읽고 Claude로 연구 필드를 추출한다."""
    text = Path(todo_path).read_text(encoding="utf-8")

    prompt = f"""아래 연구 계획 문서를 읽고 JSON으로 구조화해 주세요.
반드시 아래 8개 키를 모두 포함해야 합니다.
이미지 경로가 없으면 image_paths는 빈 배열로, image_labels도 빈 배열로 반환하세요.
데이터 경로가 없으면 data_path는 빈 문자열로 반환하세요.

출력 형식 (JSON만, 설명 없이):
{{
  "topic": "한 줄 연구 주제 (반드시 영어로 작성. 한국어 입력이면 영어로 번역)",
  "details": "구체적 내용 요약 (1~2문장)",
  "problem_definition": "문제 정의 (2~3문장)",
  "desired_outcome": "원하는 결과 (2~3문장)",
  "constraints": "제약 조건 (쉼표 구분)",
  "target_metric": "목표 지표 (쉼표 구분)",
  "data_path": "데이터 경로 (없으면 빈 문자열)",
  "image_paths": ["경로1", "경로2"],
  "image_labels": ["레이블1", "레이블2"]
}}

---
{text}
"""
    raw = query_claude(prompt)
    return parse_json(raw)


# ──────────────────────────────────────────────────────────
# 실행 함수
# ──────────────────────────────────────────────────────────

async def run_research(
    topic: str,
    details: str,
    problem_definition: str,
    desired_outcome: str,
    constraints: str = "",
    target_metric: str = "",
    data_path: str = "",
    image_paths: list[str] | None = None,
    image_labels: list[str] | None = None,
    auto_approve: bool = False,
    runner_type: str = "github",
) -> None:
    """협업 모드 연구 파이프라인을 실행한다."""

    from lab.config import workspace, reports_dir as rdir, results_dir, topic_slug as make_slug

    image_paths  = image_paths  or []
    image_labels = image_labels or []
    topic_slug   = make_slug(topic)

    # workspace 디렉토리 생성: experiments/{slug}/reports/, results/
    rdir(topic_slug).mkdir(parents=True, exist_ok=True)
    results_dir(topic_slug).mkdir(parents=True, exist_ok=True)

    # 1단계 이미지 인자 구성
    image_args = ""
    if image_paths:
        paths_str  = " ".join(f'"{p}"' for p in image_paths)
        labels_str = " ".join(f'"{l}"' for l in image_labels) if image_labels else ""
        image_args = f"\n    --image {paths_str}"
        if labels_str:
            image_args += f" \\\n    --image-labels {labels_str}"

    # ── Precondition validator 함수 (user_prompt에 주입) ──
    precondition_check = f"""
⚠️ 파이프라인 선행 조건 검증기 (자동 생성):
  topic_slug = "{topic_slug}"
  workspace  = "experiments/{topic_slug}"

각 단계 시작 전에 아래 Python 명령으로 선행 조건을 검증하세요:
  python -c "from lab.config import validate_stage_preconditions; m = validate_stage_preconditions(N, '{topic_slug}'); print('PRECONDITION OK' if not m else f'MISSING: {{m}}')"
(N을 해당 단계 번호로 치환)

검증 실패 시 해당 단계를 실행하지 말고 이전 단계를 먼저 완료하세요.
"""

    user_prompt = f"""다음 연구를 8단계 파이프라인으로 진행해주세요.

- 연구 주제:   {topic}
- 구체적 내용: {details}
- 문제 정의:   {problem_definition}
- 원하는 결과: {desired_outcome}
- 제약 조건:   {constraints if constraints else "없음"}
- 목표 지표:   {target_metric if target_metric else "미정"}
- 데이터 경로: {data_path if data_path else "미지정 (공개 데이터셋이면 자동 다운로드)"}
{"- 참조 이미지: " + ", ".join(image_paths) if image_paths else ""}

topic_slug = "{topic_slug}"
workspace  = "experiments/{topic_slug}"

1단계 실행 시 아래 명령을 사용하세요:
  python -m lab.topic_analyzer \\
    --topic "{topic}" \\
    --details "{details}" \\
    --problem-definition "{problem_definition}" \\
    --desired-outcome "{desired_outcome}" \\
    --constraints "{constraints}" \\
    --target-metric "{target_metric}"{image_args}

모든 파일은 experiments/{topic_slug}/ workspace 아래에 저장됩니다.

3단계는 반드시 --mode collaborative 옵션으로 실행하세요.
4단계는 반드시 --refine --target-score {PDF_SCORE_THRESHOLD} 옵션으로 실행하세요.
5단계 PDF는 점수 무관 항상 생성됩니다.

데이터 경로 규칙:
- 데이터 경로가 지정되면 해당 경로에 데이터가 존재하는지 먼저 확인하세요.
- 존재하면 그대로 사용: default.yaml의 data_dir에 해당 경로를 설정하세요.
- 존재하지 않고 공개 데이터셋이면 data.py에서 해당 경로에 자동 다운로드하도록 구현하세요.
- 데이터 경로가 미지정이면 data.py에서 torchvision 등으로 자동 다운로드하도록 구현하세요.

8단계 실행 시 runner_type = "{runner_type}" 을 사용하세요.
시스템 프롬프트의 {{runner_type}} 을 "{runner_type}" 으로 치환하세요.
{precondition_check}"""

    # auto_approve 모드: 5단계 자동 승인, AskUserQuestion 불필요
    if auto_approve:
        user_prompt += """

⚠️ 자동 승인 모드 (CI/CD):
  5단계 실행 시 반드시 --auto-approve 플래그를 추가하세요:
    python -m lab.user_approval \
      --topic-file      experiments/{topic_slug}/reports/topic_analysis.json \
      --hypothesis-file experiments/{topic_slug}/reports/hypothesis.json \
      --validation-file experiments/{topic_slug}/reports/validation.json \
      --papers-file     experiments/{topic_slug}/reports/papers.json \
      --auto-approve
  이 플래그가 있으면 input() 대기 없이 자동 approve 처리됩니다.
  AskUserQuestion을 사용하지 마세요."""

    _print_header(topic, image_paths)

    allowed_tools = [
        "WebSearch",
        "WebFetch",
        "Bash",
        "Read",
        "Write",
        "Edit",
    ]
    if not auto_approve:
        allowed_tools.append("AskUserQuestion")

    async for message in query(
        prompt=user_prompt,
        options=ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT.replace("8.5", str(PDF_SCORE_THRESHOLD)),
            allowed_tools=allowed_tools,
            cwd=str(Path(__file__).parent),
            max_turns=500,
        ),
    ):
        # 진행 상황 실시간 출력
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    text = block.text.strip()
                    if text:
                        # 단계 표시만 간결하게 출력
                        for marker in ["[단계", "[Round", "[협업", "[GPT", "[Gemini",
                                       "[Claude", "[점수", "[완료", "[승인"]:
                            if marker in text:
                                print(f"  {text[:120]}")
                                break

        elif isinstance(message, ResultMessage):
            _print_footer(message.result)


# ──────────────────────────────────────────────────────────
# 출력 헬퍼
# ──────────────────────────────────────────────────────────

def _print_header(topic: str, image_paths: list[str] | None = None) -> None:
    print("\n" + "=" * 60)
    print("  AI-Driven Research Automation")
    print("  협업 모드: Claude + OpenAI + Gemini")
    print("=" * 60)
    print(f"  주제: {topic}")
    if image_paths:
        print(f"  참조 이미지: {len(image_paths)}장")
        for p in image_paths:
            print(f"    - {p}")
    print("  파이프라인:")
    steps = [
        "1. 주제 분석",
        "2. 논문 검색 (최신 2년 우선)",
        "3. 협업 가설 수립 (5라운드 토론)",
        f"4. LLM 검증 + 자동 개선 (목표 {PDF_SCORE_THRESHOLD}점)",
        "5. 사용자 승인 + PDF 보고서",
        "6. GitHub 코드 분석",
        "7. PyTorch 모델 생성",
        "8. 실험 실행",
    ]
    for s in steps:
        print(f"    {s}")
    print("=" * 60 + "\n")


def _print_footer(result: str) -> None:
    print("\n" + "=" * 60)
    print("  연구 파이프라인 완료")
    print("=" * 60)
    if result:
        print(result)


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Driven Deep Learning Research Automation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--todo", metavar="FILE", default="todo.md",
        help="연구 계획을 자유 서술한 마크다운 파일 (기본값: todo.md)",
    )
    # todo.md 없이 직접 입력하는 경우 (선택)
    parser.add_argument("--topic")
    parser.add_argument("--details")
    parser.add_argument("--problem-definition")
    parser.add_argument("--desired-outcome")
    parser.add_argument("--constraints",   default="")
    parser.add_argument("--target-metric", default="")
    parser.add_argument("--data-path",     default="",
                        help="데이터 경로 (비워두면 공개 데이터셋 자동 다운로드)")
    parser.add_argument("--image", nargs="*", dest="image_paths", default=[], metavar="PATH")
    parser.add_argument("--image-labels",  nargs="*", default=[], metavar="LABEL")
    parser.add_argument("--auto-approve",  action="store_true",
                        help="5단계 사용자 승인을 자동 처리 (CI/CD용)")
    parser.add_argument("--runner-type",  default="github", choices=["local", "github"],
                        help="실험 실행 방식 (local: 직접 실행, github: Actions workflow dispatch)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # CLI 직접 입력이 없으면 todo.md에서 파싱
    if args.topic is None:
        todo_file = args.todo
        if not Path(todo_file).exists():
            raise FileNotFoundError(
                f"'{todo_file}' 파일이 없습니다. "
                "todo.md를 작성하거나 --topic 등으로 직접 입력하세요."
            )
        print(f"  todo.md 파싱 중: {todo_file}")
        fields = parse_todo(todo_file)
        topic              = fields["topic"]
        details            = fields["details"]
        problem_definition = fields["problem_definition"]
        desired_outcome    = fields["desired_outcome"]
        constraints        = fields.get("constraints", "")
        target_metric      = fields.get("target_metric", "")
        data_path          = fields.get("data_path", "")
        image_paths        = fields.get("image_paths", [])
        image_labels       = fields.get("image_labels", [])
        print(f"  파싱 완료 → 주제: {topic}\n")
    else:
        topic              = args.topic
        details            = args.details or ""
        problem_definition = args.problem_definition or ""
        desired_outcome    = args.desired_outcome or ""
        constraints        = args.constraints
        target_metric      = args.target_metric
        data_path          = args.data_path
        image_paths        = args.image_paths
        image_labels       = args.image_labels

    async def main():
        await run_research(
            topic=topic,
            details=details,
            problem_definition=problem_definition,
            desired_outcome=desired_outcome,
            constraints=constraints,
            target_metric=target_metric,
            data_path=data_path,
            image_paths=image_paths,
            image_labels=image_labels,
            auto_approve=args.auto_approve,
            runner_type=args.runner_type,
        )

    anyio.run(main)
