"""
Stage 6: GitHub 코드 분석

GitHub API로 관련 레포를 검색하고 주요 파일을 분석하여
재사용 가능한 컴포넌트와 코드 스니펫을 추출한다.
결과는 experiments/{slug}/reports/code_analysis.json에 저장한다.

개선 사항:
  - 복합 쿼리 구성 (primary + secondary + constraints + architecture 조합, 최대 5개)
  - 레포 메타데이터 확장 (pushed_at, default_branch, license)
  - 파일명 + 콘텐츠 + 구조적 코드 신호 기반 3단계 관련성 스코링
  - 핵심 코드 블록 스니펫 추출 (first-chunk 대신 class/forward/loss 우선)
  - analyzed_sources 추적성 필드 확장 (selection_signals, snippet_strategy)

사용법:
  python -m lab.code_analyzer \
    --topic-file      experiments/{slug}/reports/topic_analysis.json \
    --hypothesis-file experiments/{slug}/reports/hypothesis.json
"""

import argparse
import base64
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from lab.config import GITHUB_TOKEN, query_claude, parse_json, topic_slug as _topic_slug, reports_dir as _reports_dir


GITHUB_API = "https://api.github.com"
_HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "research-agent"}
if GITHUB_TOKEN:
    _HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def _gh_get(path: str) -> dict | list:
    url = f"{GITHUB_API}{path}"
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────
# 4.1 복합 쿼리 구성
# ──────────────────────────────────────────

def _build_search_queries(topic: dict, hypothesis: dict | None = None) -> list[str]:
    """topic 키워드에서 3~5개의 복합 검색 쿼리를 생성한다.

    우선순위:
      1. primary[0] + primary[1]              — 가장 구체적인 복합 쿼리
      2. primary[0] + secondary[0]            — 도메인 + metric/constraint 조합
      3. primary[1] + secondary[1]            — 두 번째 복합 쿼리
      4. primary[0] + constraint_tokens       — 제약 조건 포함 복합 쿼리
      5. architecture_token + primary[0]      — 가설 아키텍처 + 도메인 (가설 있을 때)

    단일 키워드 쿼리는 복합 후보가 없을 때만 fallback으로 사용한다.
    """
    inp         = topic.get("input", {})
    primary     = topic.get("search_keywords", {}).get("primary", [])
    secondary   = topic.get("search_keywords", {}).get("secondary", [])
    constraints = inp.get("constraints", "")

    queries: list[str] = []

    # 쿼리 1: primary 상위 2개 복합 (최우선)
    if len(primary) >= 2:
        queries.append(f"{primary[0]} {primary[1]}")
    elif primary:
        queries.append(primary[0])   # fallback: 단일 키워드

    # 쿼리 2: primary[0] + secondary[0] (도메인 + 구현 단서)
    if primary and secondary:
        queries.append(f"{primary[0]} {secondary[0]}")

    # 쿼리 3: primary[1] + secondary[1] — 단일 primary[2] 대신 복합 쿼리 사용
    if len(primary) >= 2 and len(secondary) >= 2:
        queries.append(f"{primary[1]} {secondary[1]}")
    elif len(primary) >= 3:
        # secondary가 부족하면 primary[0]+primary[2] 복합으로 대체
        queries.append(f"{primary[0]} {primary[2]}")

    # 쿼리 4: primary[0] + constraint tokens (제약 조건 포함)
    if constraints and primary:
        ctokens = [t for t in re.split(r"\W+", constraints) if len(t) > 3][:2]
        if ctokens:
            queries.append(f"{primary[0]} {' '.join(ctokens)}")

    # 쿼리 5: 가설 아키텍처 + primary[0] (아키텍처 힌트 활용)
    if hypothesis and primary:
        arch = hypothesis.get("experiment_plan", {}).get("architecture", "")
        arch_tokens = [t for t in re.split(r"\W+", arch) if len(t) > 4][:2]
        if arch_tokens:
            queries.append(f"{arch_tokens[0]} {primary[0]}")

    # 중복 제거, 최대 5개
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique[:5]


# ──────────────────────────────────────────
# 4.2 GitHub 레포 검색 (확장 메타데이터)
# ──────────────────────────────────────────

def search_repos(query: str, max_results: int = 5) -> list[dict]:
    """키워드로 GitHub 레포를 검색하고 확장 메타데이터를 반환한다."""
    params = urllib.parse.urlencode({
        "q": f"{query} language:python",
        "sort": "stars",
        "order": "desc",
        "per_page": max_results,
    })
    data = _gh_get(f"/search/repositories?{params}")
    if "error" in data:
        print(f"    [GitHub] 검색 실패: {data['error']}")
        return []

    return [
        {
            "full_name":      r["full_name"],
            "description":    r.get("description", ""),
            "stars":          r.get("stargazers_count", 0),
            "url":            r.get("html_url", ""),
            "topics":         r.get("topics", []),
            "pushed_at":      r.get("pushed_at", ""),
            "default_branch": r.get("default_branch", "main"),
            "license":        (r.get("license") or {}).get("spdx_id", ""),
        }
        for r in data.get("items", [])
    ]


# ──────────────────────────────────────────
# 레포 파일 목록 조회
# ──────────────────────────────────────────

def get_repo_structure(full_name: str, path: str = "") -> list[str]:
    """레포의 파일 구조를 가져온다."""
    data = _gh_get(f"/repos/{full_name}/contents/{path}")
    if isinstance(data, dict) and "error" in data:
        return []

    files = []
    if isinstance(data, list):
        for item in data:
            if item["type"] == "file":
                files.append(item["path"])
            elif item["type"] == "dir" and item["name"] not in (
                ".git", "__pycache__", "node_modules", ".github"
            ):
                sub = get_repo_structure(full_name, item["path"])
                files.extend(sub[:10])
    return files


def get_file_content(full_name: str, file_path: str, max_lines: int = 300) -> str:
    """파일 내용을 가져온다 (early+middle+late 섹션 조합, 최대 max_lines줄).

    짧은 파일(≤ max_lines)은 전체 반환.
    긴 파일은 앞(1/3) + 중간(1/3) + 뒤(1/3) 섹션을 스킵 마커와 함께 조합하여
    파일 전체에 분산된 앵커(class/forward/loss 등)를 탐색 가능하게 한다.
    """
    data = _gh_get(f"/repos/{full_name}/contents/{file_path}")
    if not (isinstance(data, dict) and data.get("encoding") == "base64"):
        return ""
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    lines   = content.splitlines()
    total   = len(lines)

    if total <= max_lines:
        return "\n".join(lines)

    section    = max_lines // 3
    early      = lines[:section]
    mid_start  = max(section, total // 2 - section // 2)
    middle     = lines[mid_start: mid_start + section]
    late_start = max(mid_start + section, total - section)
    late       = lines[late_start:]

    parts: list[str] = []
    parts.extend(early)
    if mid_start > section:
        parts.append(f"# ... [{mid_start - section} lines skipped] ...")
    parts.extend(middle)
    if late_start > mid_start + section:
        parts.append(f"# ... [{late_start - (mid_start + section)} lines skipped] ...")
    parts.extend(late)
    return "\n".join(parts)


# ──────────────────────────────────────────
# 4.3 파일 관련성 스코링 (2단계)
# ──────────────────────────────────────────

# Stage 1 코스 필터: 파일명/경로 기반
_GENERIC_FILE_KW    = ("model", "train", "network", "arch", "loss", "dataset", "utils")
# 높은 우선순위 파일명 토큰 (model/train/loss 등)
_MODEL_PREF_NAMES   = frozenset({"model", "train", "loss", "dataset", "module", "network", "arch"})
# 낮은 우선순위 파일명 (utils-only)
_UTILS_ONLY_NAMES   = frozenset({"util", "helper", "setup", "install", "readme", "config"})

# 콘텐츠 스코링 최대 문자 수 (성능)
_CONTENT_SCORE_LIMIT = 6000

# 구조적 코드 신호 (패턴, 가중치) — 실제 모델/학습 코드 감지
_CODE_STRUCT_SIGNALS: list[tuple[str, float]] = [
    (r"\bdef\s+forward\s*\(",            0.12),   # forward() 메서드
    (r"\bnn\.Module\b",                  0.10),   # nn.Module 상속
    (r"\bdef\s+training_step\s*\(",      0.10),   # Lightning training_step
    (r"\bloss\s*=\s*\w+\s*\(",           0.08),   # loss = func(...)
    (r"\bDataLoader\s*\(",               0.08),   # DataLoader 사용
    (r"\bclass\s+\w+\s*\(",              0.06),   # 클래스 정의
    (r"\bdef\s+validation_step\s*\(",    0.06),   # Lightning validation_step
    (r"\bDataset\b",                     0.05),   # Dataset 클래스
    (r"\boptim\.\w+\(",                  0.05),   # optimizer 생성
    (r"\btorch\.\w+\b|\bF\.\w+\b",       0.04),   # torch.* / F.* 호출
]
_CODE_STRUCT_MAX = 0.30   # 구조 신호 최대 기여도

# 낮은 가치 경로 패턴 (utils-only 외 추가 패널티 대상)
_LOW_VALUE_PATH_PATTERNS: list[str] = [
    r"setup\.py$", r"install\.py$", r"requirements",
    r"__init__\.py$", r"conftest\.py$", r"test_\w+\.py$",
]


def _tokenize(text: str, min_len: int = 3) -> frozenset[str]:
    """텍스트를 소문자 정규화 토큰 집합으로 변환한다.

    [\\W+]로 1차 분리 → snake_case를 _ 기준으로 2차 분리
    → CamelCase 경계를 3차 분리 후 전체 합집합을 반환한다.

    예시:
      feature_extractor  → {feature, extractor, feature_extractor}
      ResidualBlock       → {residual, block, residualblock}
      train_step          → {train, step, train_step}
    """
    tokens: set[str] = set()
    for raw in re.split(r"\W+", text):
        if not raw:
            continue
        # snake_case 분리
        snake_parts = [p for p in raw.split("_") if p]
        # CamelCase 분리 (예: ResidualBlock → [Residual, Block])
        camel_parts: list[str] = []
        for part in snake_parts:
            camel_parts.extend(re.sub(r"([a-z])([A-Z])", r"\1 \2", part).split())
        # 원본 토큰 + 분리된 부분 모두 포함
        for t in [raw] + snake_parts + camel_parts:
            tl = t.lower()
            if len(tl) >= min_len:
                tokens.add(tl)
    return frozenset(tokens)


def _get_hypothesis_tokens(hypothesis: dict) -> frozenset[str]:
    """가설에서 관련성 스코링용 토큰 집합을 추출한다."""
    hyp = hypothesis.get("hypothesis", {})
    text = " ".join([
        hyp.get("statement", ""),
        hyp.get("statement_kr", ""),
        hypothesis.get("experiment_plan", {}).get("architecture", ""),
    ])
    return _tokenize(text, min_len=4)


def _score_code_structure(content: str) -> float:
    """콘텐츠에서 구조적 코드 신호를 감지하여 점수를 반환한다 (0~_CODE_STRUCT_MAX)."""
    score = 0.0
    for pattern, weight in _CODE_STRUCT_SIGNALS:
        if re.search(pattern, content):
            score += weight
    return min(score, _CODE_STRUCT_MAX)


def _is_relevant_file(path: str, topic_keywords: tuple[str, ...] = ()) -> bool:
    """모델/학습 관련 파일인지 1차 판단 (stage 1 코스 필터)."""
    all_kw = _GENERIC_FILE_KW + tuple(kw.lower() for kw in topic_keywords)
    name = path.lower()
    return any(kw in name for kw in all_kw) and name.endswith(".py")


def _score_file_relevance(
    path: str,
    content: str,
    topic_keywords: tuple[str, ...],
    hypothesis: dict,
) -> tuple[float, dict]:
    """파일의 관련성 점수와 신호 내역을 반환한다.

    스코링 요소:
      1. 경로 토큰 ∩ 토픽 키워드 오버랩
      2. 우선 파일 유형 보너스 (model/train/loss 등)
      3. utils-only / 저가치 경로 패널티
      4. 콘텐츠 토큰 ∩ 토픽 키워드 정규화 오버랩 (주요 신호)
      4b. 약한 substring 보너스 (보조 신호)
      5. 구조적 코드 신호 (forward/training_step/nn.Module 등) (주요 신호)
      6. 콘텐츠 토큰 ∩ 가설 토큰 오버랩

    Returns:
        (score, signals) — score: 0.0~1.0, signals: 기여도 breakdown dict
    """
    score = 0.0
    stem  = Path(path).stem.lower()

    # ── 토큰 집합 사전 계산 ──────────────────────────────
    path_tokens  = _tokenize(path)
    topic_tokens = frozenset(kw.lower() for kw in topic_keywords if len(kw) >= 3)

    # 1. 경로 토큰 오버랩 (토큰 기반)
    path_overlap  = len(path_tokens & topic_tokens)
    path_contrib  = min(path_overlap * 0.15, 0.30)
    score        += path_contrib

    # 2. 우선 파일 유형 보너스 (stem 토큰 기반)
    stem_tokens     = _tokenize(stem, min_len=2)
    file_type_bonus = 0.20 if (stem_tokens & _MODEL_PREF_NAMES) else 0.0
    score          += file_type_bonus

    # 3. utils-only / 저가치 경로 패널티
    utils_penalty = 0.0
    if stem in _UTILS_ONLY_NAMES:
        utils_penalty = -0.10
    elif any(re.search(p, path.lower()) for p in _LOW_VALUE_PATH_PATTERNS):
        utils_penalty = -0.12
    score += utils_penalty

    # ── 콘텐츠 기반 스코링 ───────────────────────────────
    content_overlap = 0.0
    substr_bonus    = 0.0
    struct_score    = 0.0
    hyp_contrib     = 0.0

    if content:
        content_trunc  = content[:_CONTENT_SCORE_LIMIT]
        content_tokens = _tokenize(content_trunc)

        # 4. 콘텐츠-토픽 토큰 오버랩 (정규화, 주요 신호)
        kw_overlap      = len(content_tokens & topic_tokens)
        total_kw        = max(len(topic_tokens), 1)
        content_overlap = min((kw_overlap / total_kw) * 0.40, 0.35)
        score          += content_overlap

        # 4b. 약한 substring 보너스 (보조 신호)
        content_lower = content_trunc.lower()
        substr_hits   = sum(1 for kw in topic_keywords if kw in content_lower)
        substr_bonus  = min(substr_hits * 0.02, 0.06)
        score        += substr_bonus

        # 5. 구조적 코드 신호 (주요 신호)
        struct_score = _score_code_structure(content_trunc)
        score       += struct_score

        # 6. 가설 토큰 오버랩
        hyp_tokens  = _get_hypothesis_tokens(hypothesis)
        hyp_overlap = len(content_tokens & hyp_tokens)
        hyp_contrib = min(hyp_overlap * 0.015, 0.20)
        score      += hyp_contrib

    signals = {
        "path_overlap":       round(path_contrib, 3),
        "file_type_bonus":    round(file_type_bonus, 3),
        "utils_penalty":      round(utils_penalty, 3),
        "content_overlap":    round(content_overlap, 3),
        "code_structure":     round(struct_score, 3),
        "hypothesis_overlap": round(hyp_contrib, 3),
    }

    return min(max(score, 0.0), 1.0), signals


# 스니펫 추출용 앵커 패턴 (우선순위 순서, 레이블, 블록 크기)
# priority 0 = 최우선 (forward); 인덱스 순서가 우선순위
_SNIPPET_ANCHORS: list[tuple[str, str, int]] = [
    (r"^\s*def\s+forward\s*\(",          "forward",          40),
    (r"^\s*def\s+training_step\s*\(",    "training_step",    35),
    (r"^\s*class\s+\w+.*:",              "class_def",        60),
    (r"^\s*def\s+validation_step\s*\(",  "validation_step",  30),
    (r"^\s*class\s+\w+Dataset",          "dataset_class",    50),
    (r"DataLoader\s*\(",                 "dataloader",       25),
    (r"^\s*loss\s*=",                    "loss_def",         20),
    (r"^\s*def\s+__init__\s*\(",         "constructor",      35),
]

# 블록 내 재사용 가능성을 나타내는 코드 품질 신호
_BLOCK_QUALITY_SIGNALS: list[tuple[str, float]] = [
    (r"\bnn\.Module\b|\bnn\.\w+\(",  0.20),   # nn.Module / nn 레이어 사용
    (r"\btorch\.\w+\b|\bF\.\w+\b",   0.15),   # torch.* / F.* 호출
    (r"\bDataLoader\b|\bDataset\b",   0.15),   # DataLoader / Dataset
    (r"\boptim\.\w+\(",               0.10),   # optimizer 생성
    (r"\bloss\s*=\s*\w+\s*\(",        0.10),   # loss = func(...)
]


def _score_block(
    block_lines: list[str],
    anchor_pri: int,
    topic_tokens: frozenset[str],
) -> float:
    """앵커 블록의 품질 점수를 계산한다 (앵커 우선순위 + 코드 신호 + 토픽 오버랩)."""
    block_text = "\n".join(block_lines)
    # 앵커 우선순위 기반 기본 점수 (priority 0 = 1.0, 마지막 = ~0.12)
    base = (len(_SNIPPET_ANCHORS) - anchor_pri) / len(_SNIPPET_ANCHORS)
    # 블록 내 코드 품질 신호
    quality = sum(w for pat, w in _BLOCK_QUALITY_SIGNALS if re.search(pat, block_text))
    quality = min(quality, 0.40)
    # 토픽 토큰 오버랩 (선택적)
    overlap = 0.0
    if topic_tokens:
        block_tokens = _tokenize(block_text)
        overlap = min(len(block_tokens & topic_tokens) * 0.05, 0.20)
    return base + quality + overlap


def _extract_relevant_snippet(
    content: str,
    max_chars: int = 2000,
    topic_tokens: frozenset[str] | None = None,
) -> tuple[str, str]:
    """파일에서 모델/학습 핵심 코드 블록을 추출한다.

    각 앵커(forward/training_step/class_def 등)에 대해 블록을 수집하고
    품질 점수로 정렬하여 상위 2~3개를 max_chars 이내로 반환한다.
    적합한 앵커가 없으면 파일 앞부분(head_fallback)을 반환한다.

    Returns:
        (snippet, strategy) — strategy: 적용된 추출 전략 레이블 (machine-readable)
    """
    lines = content.splitlines()
    _topic = topic_tokens or frozenset()

    # 앵커별 후보 블록 수집 (동일 레이블 내 최고 점수 블록만 유지)
    best_per_label: dict[str, tuple[float, list[str]]] = {}
    for i, line in enumerate(lines):
        for pri, (pattern, label, block_size) in enumerate(_SNIPPET_ANCHORS):
            if re.search(pattern, line):
                block_lines = lines[max(0, i - 1): i + block_size]
                score = _score_block(block_lines, pri, _topic)
                if label not in best_per_label or score > best_per_label[label][0]:
                    best_per_label[label] = (score, block_lines)
                break  # 한 줄에 하나의 앵커만

    if not best_per_label:
        return content[:max_chars], "head_fallback"

    # 점수 내림차순 정렬 후 상위 블록 선택
    ranked = sorted(best_per_label.items(), key=lambda x: x[1][0], reverse=True)
    selected_blocks: list[str] = []
    strategy_labels: list[str] = []

    for label, (score, block_lines) in ranked:
        selected_blocks.extend(block_lines)
        selected_blocks.append("")  # 블록 구분
        strategy_labels.append(label)
        if len("\n".join(selected_blocks)) >= max_chars:
            break

    snippet  = "\n".join(selected_blocks)[:max_chars]
    strategy = "+".join(strategy_labels) if strategy_labels else "head_fallback"
    return snippet, strategy


# ──────────────────────────────────────────
# Claude로 코드 분석
# ──────────────────────────────────────────

def _analyze_code_with_claude(
    hypothesis: dict,
    repo_codes: list[dict],
) -> dict:
    """Claude로 수집된 코드를 분석하여 재사용 컴포넌트를 추출한다."""
    hyp      = hypothesis.get("hypothesis", {})
    exp_plan = hypothesis.get("experiment_plan", {})

    code_snippets = "\n\n".join([
        f"### {c['repo']} / {c['file']} [{c.get('snippet_strategy', 'head')}]\n"
        f"```python\n{c.get('snippet', c['content'][:600])}\n```"
        for c in repo_codes[:6]
    ])

    prompt = f"""당신은 딥러닝 코드 분석 전문가입니다.
아래 연구 가설과 GitHub 코드를 분석하여 재사용 가능한 컴포넌트를 추출해주세요.

## 연구 가설
{hyp.get('statement_kr', hyp.get('statement', ''))}

## 제안 아키텍처
{exp_plan.get('architecture', '')}

## GitHub 코드
{code_snippets}

## 요청
위 코드에서 연구에 활용 가능한 컴포넌트를 분석해주세요.

## 출력 형식 (반드시 아래 JSON 형식으로만 답변)
{{
  "reusable_components": [
    {{
      "name": "컴포넌트 이름",
      "type": "model|loss|dataset|trainer|utils",
      "source_repo": "레포명",
      "source_file": "파일 경로",
      "description": "설명",
      "code_snippet": "핵심 코드 (10-20줄)",
      "adaptation_needed": "수정 필요사항"
    }}
  ],
  "architecture_insights": ["아키텍처 인사이트 1", "인사이트 2"],
  "recommended_baseline": "추천 베이스라인 모델 및 이유",
  "implementation_tips": ["구현 팁 1", "팁 2"]
}}"""

    return parse_json(query_claude(prompt))


# ──────────────────────────────────────────
# 메인 분석 함수
# ──────────────────────────────────────────

def analyze_code(topic_file: str, hypothesis_file: str) -> dict:
    """
    GitHub에서 관련 코드를 수집하고 분석한다.

    Returns:
        분석 결과 dict (experiments/{slug}/reports/code_analysis.json에도 저장)
    """
    topic      = json.loads(Path(topic_file).read_text(encoding="utf-8"))
    hypothesis = json.loads(Path(hypothesis_file).read_text(encoding="utf-8"))

    topic_name = topic.get("input", {}).get("topic", "research")
    topic_slug = _topic_slug(topic_name)

    # 파일 필터 키워드 (primary + secondary 단어 추출)
    primary_kw   = topic.get("search_keywords", {}).get("primary", [])
    secondary_kw = topic.get("search_keywords", {}).get("secondary", [])
    topic_file_kw: tuple[str, ...] = tuple(
        word.lower()
        for kw in (primary_kw + secondary_kw)[:8]
        for word in kw.split()
        if len(word) > 3
    )

    # ── 4.1 복합 쿼리로 레포 검색 (가설 아키텍처 힌트 포함) ──
    queries = _build_search_queries(topic, hypothesis)
    print(f"    [GitHub] 검색 쿼리 {len(queries)}개: {queries}")

    repos: list[dict] = []
    for q in queries:
        print(f"    [GitHub] 검색: {q}")
        found = search_repos(q, max_results=4)
        repos.extend(found)
        time.sleep(1)

    # 중복 제거 (full_name 기준)
    seen: set[str] = set()
    unique_repos: list[dict] = []
    for r in repos:
        if r["full_name"] not in seen:
            seen.add(r["full_name"])
            unique_repos.append(r)

    print(f"    레포 {len(unique_repos)}개 발견")

    # ── 4.3 파일 수집 + 3단계 관련성 스코링 ──────────────
    # Stage 1: 파일명 기반 코스 필터
    # Stage 2: 콘텐츠 토큰 오버랩 + 구조적 코드 신호 스코링
    # Stage 3: 스코어 기반 상위 선택 (레포 인기도는 동점 해소에만 사용)
    # 스니펫 추출용 토픽 토큰 집합 (사전 계산)
    snippet_topic_tokens = frozenset(kw.lower() for kw in topic_file_kw if len(kw) >= 3)
    candidate_files: list[dict] = []
    for repo in unique_repos[:5]:
        fname = repo["full_name"]
        print(f"    [GitHub] 파일 분석: {fname}")
        files = get_repo_structure(fname)
        coarse = [f for f in files if _is_relevant_file(f, topic_file_kw)]
        for fpath in coarse[:6]:
            content = get_file_content(fname, fpath)
            if not content:
                continue
            # Stage 2: 토큰 + 구조 스코링 (signals 포함)
            score, signals = _score_file_relevance(fpath, content, topic_file_kw, hypothesis)
            # 핵심 코드 블록 스니펫 추출 (토픽 토큰 스코어링 포함)
            snippet, snippet_strategy = _extract_relevant_snippet(
                content, topic_tokens=snippet_topic_tokens
            )
            candidate_files.append({
                "repo":             fname,
                "file":             fpath,
                "content":          content,
                "snippet":          snippet,
                "snippet_strategy": snippet_strategy,
                "score":            round(score, 4),
                "signals":          signals,
                "stars":            repo.get("stars", 0),
                "license":          repo.get("license", ""),
                "url":              repo.get("url", ""),
            })
        time.sleep(0.5)

    # GOAL 1: 파일 품질(score) 우선 정렬; stars는 동점 해소에만 사용
    candidate_files.sort(key=lambda x: (x["score"], x["stars"] * 1e-7), reverse=True)
    repo_codes = candidate_files[:8]

    print(f"    코드 파일 {len(repo_codes)}개 수집 (스코어 기반 상위 선택)")
    if repo_codes:
        print(f"    최고 점수: {repo_codes[0]['repo']}/{repo_codes[0]['file']} "
              f"(score={repo_codes[0]['score']})")

    # ── Claude 분석 ───────────────────────────────────────
    analysis: dict = {}
    if repo_codes:
        print("    [Claude] 코드 분석 중...")
        analysis = _analyze_code_with_claude(hypothesis, repo_codes)
    else:
        analysis = {
            "reusable_components":  [],
            "architecture_insights": [],
            "recommended_baseline": "코드를 찾지 못했습니다. 직접 구현이 필요합니다.",
            "implementation_tips":  [],
        }

    # ── 4.4 analyzed_sources 추적성 필드 (GOAL 5) ────────
    def _why_selected(c: dict) -> str:
        sig = c.get("signals", {})
        parts = [f"score={c['score']:.3f}"]
        if sig.get("code_structure", 0) >= 0.10:
            parts.append("struct_code")
        if sig.get("content_overlap", 0) >= 0.15:
            parts.append("content_overlap")
        if sig.get("file_type_bonus", 0) > 0:
            parts.append("model_file")
        if sig.get("hypothesis_overlap", 0) >= 0.05:
            parts.append("hyp_match")
        return " ".join(parts)

    analyzed_sources = [
        {
            "repo":              c["repo"],
            "file":              c["file"],
            "score":             c["score"],
            "stars":             c["stars"],
            "license":           c["license"],
            "url":               c["url"],
            "why_selected":      _why_selected(c),
            "selection_signals": c.get("signals", {}),
            "snippet_strategy":  c.get("snippet_strategy", "head_fallback"),
        }
        for c in repo_codes
    ]

    result = {
        "timestamp":        datetime.now().isoformat(),
        "topic":            topic_name,
        "repos_found":      [r["full_name"] for r in unique_repos],
        "repos_metadata":   [
            {k: r[k] for k in ("full_name", "stars", "license", "pushed_at", "url")}
            for r in unique_repos
        ],
        "files_analyzed":   len(repo_codes),
        "analyzed_sources": analyzed_sources,
        **analysis,
    }

    output_path = _reports_dir(topic_slug) / "code_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    코드 분석 저장: {output_path}")

    return result


def print_analysis(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  코드 분석 결과: {result['topic']}")
    print(f"{'='*60}")
    print(f"\n발견 레포: {', '.join(result.get('repos_found', []))}")
    print(f"분석 파일 수: {result.get('files_analyzed', 0)}")

    sources = result.get("analyzed_sources", [])
    if sources:
        print(f"\n[분석 소스 (상위 {len(sources)}개)]")
        for s in sources[:5]:
            print(f"  - {s['repo']}/{s['file']} "
                  f"(score={s['score']}, stars={s['stars']}, {s['why_selected']})")

    print(f"\n[추천 베이스라인]\n  {result.get('recommended_baseline', '')}")

    components = result.get("reusable_components", [])
    if components:
        print(f"\n[재사용 컴포넌트 ({len(components)}개)]")
        for c in components:
            print(f"  - [{c['type']}] {c['name']}: {c['description']}")

    tips = result.get("implementation_tips", [])
    if tips:
        print("\n[구현 팁]")
        for t in tips:
            print(f"  - {t}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub 코드 분석")
    parser.add_argument("--topic-file",      required=True)
    parser.add_argument("--hypothesis-file", required=True)
    args = parser.parse_args()

    result = analyze_code(args.topic_file, args.hypothesis_file)
    print_analysis(result)
