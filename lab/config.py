from dotenv import load_dotenv
import anyio
import json
import os
import re
from pathlib import Path
from claude_agent_sdk import query as _sdk_query, ClaudeAgentOptions, AssistantMessage

load_dotenv()

# LLM 모델 설정
CLAUDE_MODEL = "claude-opus-4-6"
OPENAI_MODEL = "gpt-5.1"   #5.4
GEMINI_MODEL = "gemini-2.5-pro" #3.1-pro

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN")
S2_API_KEY     = os.getenv("S2_API_KEY", "")

# 검증 통과 기준 (validator + PDF 공용)
SCORE_THRESHOLD = 8.5

# ──────────────────────────────────────────────────────────
# Diagnostic Engine — 통계 상수
#
# 각 임계값은 근거(rationale)를 주석으로 명시한다.
# 변경 시 근거도 함께 갱신할 것.
# ──────────────────────────────────────────────────────────

# A-2: Training Stability
# Convergence: 마지막 25% 구간의 상대 변화가 1% 미만이면 수렴으로 판정
# Rationale: 딥러닝 학습 곡선은 exponential decay 형태이며, tail 25%에서
#            1% 이하 변화는 실질적 수렴을 의미 (Goodfellow et al., Deep Learning, Ch.8)
CONVERGENCE_TAIL_FRACTION = 0.25
CONVERGENCE_REL_THRESHOLD = 0.01

# Plateau: 연속 N 에포크에서 상대 변화 < threshold이면 정체 판정
# Rationale: 3 에포크 연속 0.5% 미만 변화는 학습률 조정 또는 구조 변경이 필요한 정체 상태
PLATEAU_WINDOW = 3
PLATEAU_REL_THRESHOLD = 0.005

# Overfitting: train-val 간 상대 격차가 threshold 초과 시 과적합 의심
# Rationale: 15% gap은 regularization 부족의 일반적 기준 (dropout, weight decay 논문 참고)
OVERFIT_GAP_THRESHOLD = 0.15

# Undertraining: 마지막 에포크에서 여전히 빠르게 개선 중이면 학습 부족
# convergence threshold의 5배 이상 개선 = 아직 학습이 진행 중
UNDERTRAINING_IMPROVEMENT_FACTOR = 5

# Divergence: 최종 loss가 초기 loss의 N배 이상이면 발산
DIVERGENCE_LOSS_RATIO = 2.0

# A-3: Bottleneck 판정
# Attainment ratio < N이면 실행 실패 수준으로 판정
ATTAINMENT_FAILURE_RATIO = 0.5

# A-4: Path escalation
# Path B: 연속 plateau가 N회 이상이면 가설 수정 필요
PATH_B_CONSECUTIVE_PLATEAUS = 2
# Path C: 메커니즘 구현 후 연속 N회 이상 목표 미달 + 안정적 학습이면 가설 기각 검토
PATH_C_MIN_CONTRADICTIONS = 3
PATH_C_MIN_RUNS = 3

# A-5: Confidence model weights
CONFIDENCE_WEIGHTS = {
    "metric":         0.4,
    "stability":      0.2,
    "implementation": 0.2,
    "trend":          0.2,
}

# A-6: Ablation — 효과 크기 판정
# 상대 변화 > N이면 의미 있는 효과로 판정
ABLATION_EFFECT_THRESHOLD = 0.01  # 1% relative change

# Effect size 해석 기준 (Cohen's d analog for single-run)
# small: 0.01-0.05, medium: 0.05-0.15, large: >0.15
EFFECT_SIZE_SMALL = 0.01
EFFECT_SIZE_MEDIUM = 0.05
EFFECT_SIZE_LARGE = 0.15

# Metric value range validation — 도메인별 합리적 범위
# 범위 밖의 값은 파싱 오류로 의심
METRIC_VALID_RANGES = {
    "accuracy":     (0.0, 1.0),
    "f1":           (0.0, 1.0),
    "f1_macro":     (0.0, 1.0),
    "f1_micro":     (0.0, 1.0),
    "precision":    (0.0, 1.0),
    "recall":       (0.0, 1.0),
    "auc":          (0.0, 1.0),
    "auroc":        (0.0, 1.0),
    "ssim":         (0.0, 1.0),
    "psnr":         (0.0, 80.0),
    "loss":         (0.0, 1e6),
    "mse":          (0.0, 1e6),
    "mae":          (0.0, 1e6),
    "rmse":         (0.0, 1e6),
    "iou":          (0.0, 1.0),
    "dice":         (0.0, 1.0),
    "map":          (0.0, 1.0),
}

# ──────────────────────────────────────────────────────────
# Evidence Coverage Slots — 파이프라인 공통 계약
# topic_analyzer, paper_researcher, hypothesis_generator, hypothesis_validator 공통
# ──────────────────────────────────────────────────────────

EVIDENCE_COVERAGE_SLOTS = [
    "key_innovation",
    "expected_mechanism",
    "closest_prior_art",
    "baseline_models",
    "evaluation_metrics",
    "constraints",
    "falsification_criteria",
    "failure_modes",
    "deployment_constraints",
]

# Evidence roles — paper_researcher가 각 논문에 부여하는 역할
EVIDENCE_ROLES = {
    "closest_prior_art", "supporting_mechanism", "baseline_reference",
    "evaluation_reference", "constraint_reference",
    "failure_mode_reference", "falsification_reference",
}

# Coverage group 분류 — sufficiency 판정 기준
COVERAGE_GROUPS = {
    "novelty": {"closest_prior_art", "key_innovation", "expected_mechanism"},
    "validity": {"baseline_models", "evaluation_metrics", "falsification_criteria"},
    "feasibility": {"constraints", "deployment_constraints"},
}




# ──────────────────────────────────────────────────────────
# LLM Query 안정성
# ──────────────────────────────────────────────────────────

LLM_QUERY_TIMEOUT = 300  # 초 (5분). LLM API 응답 대기 최대 시간
LLM_QUERY_MAX_RETRIES = 2  # 실패 시 재시도 횟수

# ──────────────────────────────────────────────────────────
# Stage Dependency Map — 단계 간 필수 파일 계약
#
# 각 단계가 시작되기 전에 존재해야 하는 파일 목록.
# {slug}는 런타임에 치환된다.
# ──────────────────────────────────────────────────────────

STAGE_PRECONDITIONS = {
    1: [],  # Stage 1: 입력만 필요
    2: ["experiments/{slug}/reports/topic_analysis.json"],
    3: ["experiments/{slug}/reports/topic_analysis.json",
        "experiments/{slug}/reports/papers.json"],
    4: ["experiments/{slug}/reports/hypothesis.json"],
    5: ["experiments/{slug}/reports/hypothesis.json",
        "experiments/{slug}/reports/validation.json"],
    6: ["experiments/{slug}/reports/topic_analysis.json",
        "experiments/{slug}/reports/hypothesis.json",
        "experiments/{slug}/reports/approval.json"],
    7: ["experiments/{slug}/reports/topic_analysis.json",
        "experiments/{slug}/reports/hypothesis.json",
        "experiments/{slug}/reports/code_analysis.json"],
    8: ["experiments/{slug}/runs/v1/experiment_spec.json"],
}


def validate_stage_preconditions(stage: int, slug: str) -> list[str]:
    """주어진 단계의 선행 조건 파일들이 존재하는지 검증한다.

    Returns: list of missing file paths (empty if all ok).
    """
    missing = []
    for pattern in STAGE_PRECONDITIONS.get(stage, []):
        path = Path(pattern.replace("{slug}", slug))
        if not path.exists():
            missing.append(str(path))
    return missing


# ──────────────────────────────────────────────────────────
# Topic workspace 경로 헬퍼
# ──────────────────────────────────────────────────────────

def topic_slug(topic: str) -> str:
    """연구 주제 문자열 → 파일 경로에 사용 가능한 slug."""
    return re.sub(r"\W+", "_", topic.lower())[:30]

def workspace(slug: str) -> Path:
    """experiments/{slug}/"""
    return Path("experiments") / slug

def reports_dir(slug: str) -> Path:
    """experiments/{slug}/reports/"""
    return workspace(slug) / "reports"

def results_dir(slug: str) -> Path:
    """experiments/{slug}/results/"""
    return workspace(slug) / "results"

def run_dir(slug: str, version: int) -> Path:
    """experiments/{slug}/runs/v{N}/"""
    return workspace(slug) / "runs" / f"v{version}"

def result_version_dir(slug: str, version: int) -> Path:
    """experiments/{slug}/results/v{N}/"""
    return results_dir(slug) / f"v{version}"


def slug_from_pkg(pkg_dir: Path) -> str:
    """runs/vN → experiments/{slug}/runs/vN 에서 slug 추출."""
    # pkg_dir = experiments/{slug}/runs/vN
    return pkg_dir.parent.parent.name


def version_from_pkg(pkg_dir: Path) -> int:
    """runs/vN 에서 version 정수 추출. 실패 시 ValueError."""
    m = re.match(r"v(\d+)", pkg_dir.name)
    if not m:
        raise ValueError(f"Invalid pkg_dir name (expected vN): {pkg_dir.name}")
    return int(m.group(1))


def get_openai_client():
    """OpenAI 클라이언트를 반환한다. timeout 설정 포함."""
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_QUERY_TIMEOUT)


def get_gemini_model(model_name: str = GEMINI_MODEL):
    """Gemini GenerativeModel을 반환한다."""
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(model_name)


def llm_retry(func, *args, max_retries: int = LLM_QUERY_MAX_RETRIES, label: str = "", **kwargs):
    """LLM 호출을 retry로 감싼다. 실패 시 max_retries만큼 재시도.

    Usage:
        result = llm_retry(client.chat.completions.create, messages=[...], label="GPT interpret")
    """
    import time
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s...
                print(f"    [{label or 'LLM'}] 재시도 {attempt+1}/{max_retries} ({wait}s 대기): {e}")
                time.sleep(wait)
    raise last_error  # type: ignore[misc]


def prompt_hash(prompt: str) -> str:
    """프롬프트의 SHA-256 해시 (처음 12자)를 반환한다. 재현성 추적용."""
    import hashlib
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def query_claude(prompt: str, image_paths: list[str] | None = None,
                 timeout: int | None = None) -> str:
    """claude_agent_sdk를 통해 Claude에 단일 쿼리를 실행한다 (API 키 불필요).

    image_paths를 전달하면 Read 툴을 허용하여 Claude가 이미지를 직접 읽는다.
    timeout: 초 단위 타임아웃 (기본값: LLM_QUERY_TIMEOUT)
    """
    _timeout = timeout if timeout is not None else LLM_QUERY_TIMEOUT

    async def _run() -> str:
        import asyncio

        full_prompt = prompt
        if image_paths:
            paths = "\n".join(f"- {p}" for p in image_paths)
            full_prompt += f"\n\n분석할 이미지 파일:\n{paths}"

        tools     = ["Read"] if image_paths else []
        max_turns = max(3, len(image_paths) + 2) if image_paths else 1

        async def _query_inner() -> str:
            text = ""
            async for msg in _sdk_query(
                prompt=full_prompt,
                options=ClaudeAgentOptions(
                    model=CLAUDE_MODEL,
                    allowed_tools=tools,
                    max_turns=max_turns,
                ),
            ):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "text"):
                            text += block.text
            return text

        # timeout 적용
        try:
            return await asyncio.wait_for(_query_inner(), timeout=_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Claude query timed out after {_timeout}s. "
                f"Prompt length: {len(prompt)} chars."
            )

    return anyio.run(_run)


def parse_json(text: str) -> dict:
    """LLM 응답에서 마크다운 펜스를 제거하고 JSON을 파싱한다."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # ```json ... ``` 형식 처리
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text and not text.startswith("{") and not text.startswith("["):
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)
