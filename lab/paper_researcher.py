"""
Stage 2: 논문 조사

arXiv API와 Semantic Scholar API로 최신 관련 논문을 검색한다.
- 첫 시도: 최근 2년치 (current_year-1 ~ current_year)
- 부족하면 1년씩 소급하여 최대 MAX_YEARS_BACK년까지 확장
결과는 reports/papers_{topic}.json에 저장한다.

사용법:
  python -m lab.paper_researcher --topic-file reports/topic_analysis.json
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from lab.config import GITHUB_TOKEN, S2_API_KEY


ARXIV_API = "https://export.arxiv.org/api/query"
S2_API    = "https://api.semanticscholar.org/graph/v1"

MIN_PAPERS    = 10   # 이 수 이상이면 연도 확장 중단
MAX_YEARS_BACK = 6   # 최대 소급 연수 (2년 기본 + 최대 4년 추가)
INIT_WINDOW   = 2    # 첫 검색 연도 윈도우 (현재 연도 포함 2년)

# 학회 논문 검색 대상 (S2 venue 필드 부분 매칭)
CONF_VENUES = ["AAAI", "CVPR", "ICCV", "ECCV", "NeurIPS", "MICCAI", "ICLR", "ICML"]

TOP_FULLTEXT      = 8     # 본문 섹션을 읽을 최대 논문 수
MAX_SECTION_CHARS = 1200  # 섹션당 최대 문자 수 (introduction/method/experiment/limitation)

# 섹션 헤더 인식 키워드 (대소문자 무관)
_SECTION_PATTERNS = {
    "introduction": re.compile(r"(?i)\b(introduction|background|motivation)\b"),
    "method":       re.compile(r"(?i)\b(method(?:ology)?|approach|model|architecture|framework|proposed)\b"),
    "experiment":   re.compile(r"(?i)\b(experiment|evaluation|result|benchmark|ablation)\b"),
    "limitation":   re.compile(r"(?i)\b(limitation|discussion|future\s+work)\b"),
}


# arXiv ti: 검색에서 제외할 generic/stopword 목록 (anchor에 엄격 적용)
_STOP_WORDS = {
    "deep", "learning", "neural", "network", "based", "using", "with", "for",
    "the", "and", "via", "model", "method", "approach", "image", "images",
    "data", "dataset", "training", "framework", "efficient", "novel",
}

# abs: 검색에서도 제외할 broad blocklist (anchor보다 완화, 하지만 generic-only 방지)
_BROAD_BLOCKLIST = _STOP_WORDS | {
    "novel", "efficient", "improved", "new", "proposed", "toward", "towards",
    "high", "low", "fast", "large", "small", "multi", "self", "end",
}

# broad fallback에서도 여전히 제외할 너무 일반적인 단어
_SOFT_FALLBACK_BLOCKLIST = {
    "model", "method", "approach", "framework", "image", "images", "data"
}


def _norm_token(t: str) -> str:
    """소문자 변환 + 복수형 's' 제거."""
    return t.lower().rstrip("s")


def _valid_query_tokens(text: str) -> list[str]:
    """길이 > 2인 토큰 추출 (구두점 분리)."""
    return [t for t in re.split(r"\W+", text or "") if len(t) > 2]


def _select_anchor_terms(query: str) -> tuple[list[str], list[str]]:
    """query에서 ti: anchor 토큰(최대 2개)과 abs: broad 토큰(최대 4개)을 선정한다.

    - 하이픈 복합어(X-ray, low-dose)는 분리하지 않고 그대로 anchor 우선 후보
    - _STOP_WORDS: anchor에서 제외
    - _BROAD_BLOCKLIST: broad에서도 제외 (단, broad는 완화 적용)
    - fallback 시 _SOFT_FALLBACK_BLOCKLIST도 함께 제외해 너무 일반적인 단어 방지
    - 길이가 길수록 구별력이 높다고 간주
    """
    # 공백 단위 분리 → 하이픈 복합어 보존, 나머지 구두점 제거
    raw_tokens = re.split(r"\s+", query.strip())
    tokens = [re.sub(r"[^\w\-]", "", t) for t in raw_tokens if re.sub(r"[^\w\-]", "", t)]
    tokens = [t for t in tokens if len(t) > 2]

    # anchor: _STOP_WORDS 완전 제외, 길이 내림차순
    specific = [t for t in tokens if _norm_token(t) not in _STOP_WORDS]
    anchor_terms = sorted(specific, key=len, reverse=True)[:2]

    # broad: _BROAD_BLOCKLIST 제외
    broad_candidates = [t for t in tokens if _norm_token(t) not in _BROAD_BLOCKLIST]

    # soft fallback: specific에서도 _SOFT_FALLBACK_BLOCKLIST 제외
    soft_fallback = [t for t in specific if _norm_token(t) not in _SOFT_FALLBACK_BLOCKLIST]

    if len(broad_candidates) < 2:
        broad_candidates = list(dict.fromkeys(broad_candidates + soft_fallback))

    broad_terms = broad_candidates[:4]  # 6 → 4

    # 최소 fallback: 둘 다 비면 원래 tokens 앞부분 사용
    if not anchor_terms:
        anchor_terms = tokens[:1]
    if not broad_terms:
        broad_terms = soft_fallback[:2] or tokens[:2]

    return anchor_terms, broad_terms


def _make_paper_id(prefix: str, raw_id: str, title: str) -> str:
    """논문 고유 ID를 생성한다. raw_id가 있으면 prefix:raw_id, 없으면 title 슬러그 기반."""
    if raw_id:
        return f"{prefix}:{raw_id}"
    slug = re.sub(r"\W+", "_", title.lower()).strip("_")[:48]
    return f"{prefix}:{slug}"


def _http_get(url: str, headers: dict = None) -> str:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


def _s2_http_get(url: str, max_retries: int = 3) -> str:
    """Semantic Scholar API GET 요청. S2_API_KEY 헤더 포함 + 429 백오프."""
    headers: dict[str, str] = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    backoff = 3  # 초
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = backoff * (2 ** attempt)
                print(f"    [S2] 429 Too Many Requests → {wait}초 대기 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"[S2] {max_retries}회 재시도 후도 429 에러: {url}")


# ──────────────────────────────────────────
# arXiv 검색 (연도 범위 지정 가능)
# ──────────────────────────────────────────

def search_arxiv(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int = 10,
) -> list[dict]:
    """arXiv에서 논문을 검색한다. year_from/year_to로 날짜 필터 적용."""
    # all: 필드는 날짜 필터와 함께 사용 시 관련 없는 결과를 반환하는 버그 존재.
    # ti: / abs: 필드를 분리하여 사용하되, 핵심 키워드만 추출해 쿼리를 구성한다.
    # _select_anchor_terms()로 도메인 특화 anchor 선정 (generic 단어 제외)
    anchor_terms, broad_terms = _select_anchor_terms(query)
    ti_part  = " AND ".join(f"ti:{t}"  for t in anchor_terms) if anchor_terms else ""
    # informative_broad: _SOFT_FALLBACK_BLOCKLIST까지 제거한 뒤 AND/OR 결정
    informative_broad = [t for t in broad_terms if _norm_token(t) not in _SOFT_FALLBACK_BLOCKLIST]
    if len(informative_broad) <= 1:
        abs_part = " OR ".join(f"abs:{t}" for t in broad_terms) if broad_terms else ""
    else:
        abs_part = " AND ".join(f"abs:{t}" for t in broad_terms)
    if ti_part and abs_part:
        kw_part = f"({ti_part}) AND ({abs_part})"
    elif ti_part:
        kw_part = f"({ti_part})"
    else:
        kw_part = f"({abs_part})" if abs_part else query

    print(f"    [arXiv query] anchor={anchor_terms} broad={broad_terms} informative={informative_broad}")
    if year_from and year_to:
        date_filter = f"submittedDate:[{year_from}0101 TO {year_to}1231]"
        search_q = f"({kw_part}) AND {date_filter}"
    else:
        search_q = kw_part

    # sortBy:submittedDate는 arXiv API 자체 버그로 엉뚱한 논문을 반환하는 경우가 있음.
    # 정렬 파라미터 제거 후 Python 레벨에서 published 기준 최신순 정렬.
    params = urllib.parse.urlencode({
        "search_query": search_q,
        "start": 0,
        "max_results": max_results,
    })
    url = f"{ARXIV_API}?{params}"

    try:
        xml = _http_get(url)
    except Exception as e:
        print(f"    [arXiv] 요청 실패: {e}")
        return []

    papers = []
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
    for entry in entries:
        def tag(t, _e=entry):
            m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", _e, re.DOTALL)
            return m.group(1).strip() if m else ""

        arxiv_id = re.search(r"<id>(.*?)</id>", entry)
        arxiv_id = arxiv_id.group(1).strip() if arxiv_id else ""
        authors   = re.findall(r"<name>(.*?)</name>", entry)
        published = tag("published")[:10]
        title     = tag("title").replace("\n", " ")
        raw_arxiv = arxiv_id.split("/abs/")[-1] if "/abs/" in arxiv_id else ""
        raw_arxiv = re.sub(r"v\d+$", "", raw_arxiv)  # 버전 suffix 제거 (예: 2204.04524v2 → 2204.04524)

        papers.append({
            "paper_id":  _make_paper_id("arxiv", raw_arxiv, title),
            "title":     title,
            "authors":   authors[:5],
            "year":      published[:4],
            "published": published,
            "abstract":  tag("summary").replace("\n", " ")[:500],
            "url":       arxiv_id,
            "arxiv_id":  raw_arxiv,
            "github":    "",
            "sections":  {},
            "source":    "arXiv",
        })

    # arXiv API sortBy 버그 우회: Python 레벨에서 published 기준 최신순 정렬
    papers.sort(key=lambda p: p.get("published") or "", reverse=True)
    return papers


# ──────────────────────────────────────────
# Semantic Scholar 검색 (연도 범위 지정 가능)
# ──────────────────────────────────────────

def search_semantic_scholar(
    query: str,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int = 10,
) -> list[dict]:
    """Semantic Scholar에서 논문을 검색한다."""
    fields = "paperId,title,authors,year,abstract,externalIds,venue"
    qs: dict = {
        "query":  query,
        "limit":  max_results,
        "fields": fields,
    }
    if year_from and year_to:
        qs["year"] = f"{year_from}-{year_to}"

    url = f"{S2_API}/paper/search?{urllib.parse.urlencode(qs)}"

    try:
        data = json.loads(_s2_http_get(url))
    except Exception as e:
        print(f"    [S2] 요청 실패: {e}")
        return []

    papers = []
    for p in data.get("data", []):
        authors  = [a.get("name", "") for a in p.get("authors", [])[:5]]
        ext      = p.get("externalIds", {})
        arxiv_id = ext.get("ArXiv", "")
        url_link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""

        title = p.get("title", "")
        papers.append({
            "paper_id":  p.get("paperId") or _make_paper_id("s2", "", title),
            "title":     title,
            "authors":   authors,
            "year":      str(p.get("year", "")),
            "published": str(p.get("year", "")),
            "abstract":  (p.get("abstract") or "")[:500],
            "url":       url_link,
            "arxiv_id":  arxiv_id,
            "github":    "",
            "sections":  {},
            "venue":     p.get("venue", ""),
            "source":    "SemanticScholar",
        })

    return papers


# ──────────────────────────────────────────
# Citation Expansion (backward / forward / similar)
# ──────────────────────────────────────────

def _s2_get_json(path: str) -> dict | list:
    """Semantic Scholar API GET 요청 (S2_API_KEY 헤더 + 429 백오프 포함)."""
    url = f"{S2_API}{path}"
    try:
        return json.loads(_s2_http_get(url))
    except Exception as e:
        print(f"    [S2] 요청 실패: {e}")
        return {}


def _get_paper_id(paper: dict) -> str:
    """Semantic Scholar에서 사용할 paper ID를 추출한다."""
    if paper.get("arxiv_id"):
        return f"ArXiv:{paper['arxiv_id']}"
    # title 기반 fallback
    return ""


def _parse_s2_paper(p: dict) -> dict | None:
    """S2 citation 결과를 표준 형식으로 변환한다."""
    if not p or not p.get("title"):
        return None
    ext = p.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv", "")
    authors = [a.get("name", "") for a in p.get("authors", [])[:5]]
    title = p["title"]
    return {
        "paper_id":  p.get("paperId") or _make_paper_id("s2", "", title),
        "title":     title,
        "authors":   authors,
        "year":      str(p.get("year") or ""),
        "published": str(p.get("year") or ""),
        "abstract":  (p.get("abstract") or "")[:500],
        "url":       f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
        "arxiv_id":  arxiv_id,
        "github":    "",
        "sections":  {},
        "venue":     p.get("venue", ""),
        "source":    "S2-citation",
    }


def expand_citations(
    seed_papers: list[dict],
    max_seeds: int = 5,
    max_per_seed: int = 5,
) -> list[dict]:
    """상위 seed 논문의 backward/forward citation + recommendations를 수집한다."""
    expanded = []
    fields = "title,authors,year,abstract,externalIds,venue"

    seeds = [p for p in seed_papers if _get_paper_id(p)][:max_seeds]
    if not seeds:
        print("    [Citation] seed 논문에 paper ID 없음 → 건너뜀")
        return []

    for sp in seeds:
        pid = _get_paper_id(sp)
        print(f"    [Citation] {sp['title'][:50]}...")

        # Backward (references)
        refs = _s2_get_json(f"/paper/{pid}/references?fields={fields}&limit={max_per_seed}")
        for item in (refs.get("data") or [])[:max_per_seed]:
            parsed = _parse_s2_paper(item.get("citedPaper", {}))
            if parsed:
                parsed["source"] = "S2-backward"
                expanded.append(parsed)

        time.sleep(0.5)

        # Forward (citations)
        cits = _s2_get_json(f"/paper/{pid}/citations?fields={fields}&limit={max_per_seed}")
        for item in (cits.get("data") or [])[:max_per_seed]:
            parsed = _parse_s2_paper(item.get("citingPaper", {}))
            if parsed:
                parsed["source"] = "S2-forward"
                expanded.append(parsed)

        time.sleep(0.5)

    print(f"    [Citation] {len(expanded)}편 수집 (backward + forward)")
    return expanded


def fetch_foundational_papers(
    seed_papers: list[dict],
    max_seeds: int = 5,
    min_co_citations: int = 2,
    max_results: int = 10,
) -> list[dict]:
    """여러 seed 논문이 공통으로 인용하는 foundational 논문을 수집한다.

    min_co_citations개 이상의 seed가 인용한 논문을 foundational로 간주하며,
    연도 제한 없이 수집한다 (foundational 논문은 오래된 경우가 많음).
    """
    fields = "title,authors,year,abstract,externalIds,venue,citationCount"
    ref_counter: dict[str, dict] = {}  # paperId → {count, paper}

    seeds = [p for p in seed_papers if _get_paper_id(p)][:max_seeds]
    if not seeds:
        print("    [Foundational] seed 논문에 paper ID 없음 → 건너뜀")
        return []

    for sp in seeds:
        pid = _get_paper_id(sp)
        print(f"    [Foundational] references 수집: {sp['title'][:50]}...")
        refs = _s2_get_json(f"/paper/{pid}/references?fields={fields}&limit=20")
        for item in (refs.get("data") or []):
            cited = item.get("citedPaper") or {}
            s2id  = cited.get("paperId", "")
            if not s2id:
                continue
            if s2id in ref_counter:
                ref_counter[s2id]["count"] += 1
            else:
                ref_counter[s2id] = {"count": 1, "paper": cited}
        time.sleep(0.5)

    # min_co_citations 이상 공통 인용된 논문만 추출 → citation count 내림차순
    candidates = [v for v in ref_counter.values() if v["count"] >= min_co_citations]
    candidates.sort(key=lambda x: x["paper"].get("citationCount") or 0, reverse=True)

    results = []
    for item in candidates[:max_results]:
        parsed = _parse_s2_paper(item["paper"])
        if parsed:
            parsed["source"] = "S2-foundational"
            parsed["co_citation_count"] = item["count"]
            results.append(parsed)

    print(f"    [Foundational] {len(results)}편 발견 (≥{min_co_citations}개 seed 공통 인용)")
    return results


def fetch_arxiv_sections(arxiv_id: str) -> dict:
    """arXiv HTML에서 Introduction/Method/Experiment/Limitation 섹션 텍스트를 추출한다.

    arXiv HTML 버전이 없거나 파싱에 실패하면 빈 딕셔너리를 반환한다.
    """
    if not arxiv_id:
        return {}
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        html = _http_get(url)
    except Exception:
        return {}

    # script/style 제거 후 태그 제거 → 순수 텍스트
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>",  " ", text,  flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 섹션 헤더 위치 수집: "1 Introduction", "2. Method", "3 Experiments" 등
    header_pat = re.compile(r"\b(\d+\.?\s+[A-Z][A-Za-z ]{3,50}?)(?=\s{2,}|\s[A-Z\d])")
    found_headers: list[tuple[int, str, str]] = []  # (pos, key, raw_name)
    for m in header_pat.finditer(text):
        raw = m.group(1).strip()
        for key, kw_pat in _SECTION_PATTERNS.items():
            if kw_pat.search(raw) and key not in {k for _, k, _ in found_headers}:
                found_headers.append((m.start(), key, raw))
                break

    sections: dict[str, str] = {}
    for idx, (pos, key, raw) in enumerate(found_headers):
        start = pos + len(raw)
        end = (found_headers[idx + 1][0]
               if idx + 1 < len(found_headers)
               else start + MAX_SECTION_CHARS * 2)
        sections[key] = text[start:end].strip()[:MAX_SECTION_CHARS]

    return sections


def _extract_github(abstract: str) -> str:
    m = re.search(r"https?://github\.com/[\w\-]+/[\w\-\.]+", abstract)
    return m.group(0) if m else ""


def _deduplicate(papers: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for p in papers:
        key = re.sub(r"\s+", " ", p["title"].lower().strip())
        if key not in seen:
            seen.add(key); unique.append(p)
    return unique


# ──────────────────────────────────────────
# 복합 질의 생성
# ──────────────────────────────────────────

def _build_search_queries(topic_data: dict) -> list[str]:
    """topic_data에서 복합 검색 질의 목록을 생성한다.

    우선순위: problem_definition + constraints 조합 질의 > details + metric > primary 키워드
    개별 키워드 나열보다 구체 phrase 조합을 앞에 배치하여 검색 정밀도를 높인다.
    """
    inp       = topic_data.get("input", {})
    keywords  = topic_data.get("search_keywords", {})
    primary   = keywords.get("primary", [])
    secondary = keywords.get("secondary", [])
    pa        = topic_data.get("problem_analysis", {})

    def key_tokens(text: str, n: int = 3) -> str:
        tokens = [t for t in re.split(r"\W+", text) if len(t) > 3
                  and t.lower() not in _STOP_WORDS]
        return " ".join(tokens[:n])

    queries: list[str] = []

    # 1순위: problem_definition + constraints 조합 (최소 1개 보장)
    prob_key = key_tokens(inp.get("problem_definition", ""), 3)
    cons_key = key_tokens(inp.get("constraints", ""), 2)
    if prob_key and cons_key:
        queries.append(f"{prob_key} {cons_key}")
    elif prob_key:
        queries.append(prob_key)

    # 2순위: details + target_metric 조합 (최소 1개 보장)
    det_key  = key_tokens(inp.get("details", ""), 3)
    met_key  = key_tokens(inp.get("target_metric", ""), 2)
    core_key = key_tokens(pa.get("core_problem", ""), 3)
    if det_key and met_key:
        queries.append(f"{det_key} {met_key}")
    elif core_key and met_key:
        queries.append(f"{core_key} {met_key}")
    elif det_key:
        queries.append(det_key)

    # 3순위: primary 키워드 (최대 2개 — 보조로 사용, 앞 복합 질의와 토큰 중복 제외)
    existing_tokens = set(re.split(r"\W+", " ".join(queries).lower()))
    for kw in primary[:2]:
        kw_tokens = set(re.split(r"\W+", kw.lower()))
        # 앞 질의와 80% 이상 토큰이 겹치면 중복으로 간주
        if kw not in queries and len(kw_tokens - existing_tokens) >= max(1, len(kw_tokens) * 0.2):
            queries.append(kw)
            existing_tokens |= kw_tokens

    # 4순위: secondary 키워드 (최대 1개 — broad blocklist 성격 항목 제외)
    def _is_broad_only(kw: str) -> bool:
        tokens = {t.lower().rstrip("s") for t in re.split(r"\W+", kw) if len(t) > 2}
        return bool(tokens) and tokens.issubset(_BROAD_BLOCKLIST)

    for kw in secondary[:3]:  # 최대 3개 후보에서 통과하는 1개 선택
        if not _is_broad_only(kw) and kw not in queries:
            queries.append(kw)
            break

    print(f"  [검색 질의 우선순위] {queries}")
    return queries


# ──────────────────────────────────────────
# 연도 범위 점진 확장 검색
# ──────────────────────────────────────────

def _search_window(
    queries: list[str],
    year_from: int,
    year_to: int,
) -> list[dict]:
    """주어진 연도 구간에서 복합 질의로 논문을 검색한다."""
    collected = []

    arxiv_qs = queries[:3]
    s2_qs    = queries[:4]
    print(f"    [_search_window] arXiv 질의 {len(arxiv_qs)}개 / S2 질의 {len(s2_qs)}개")

    for q in arxiv_qs:
        print(f"    [arXiv {year_from}-{year_to}] {q}")
        papers = search_arxiv(q, year_from, year_to, max_results=8)
        print(f"      → {len(papers)}편 수집")
        collected.extend(papers)
        time.sleep(1)

    for q in s2_qs:
        print(f"    [S2 {year_from}-{year_to}] {q}")
        papers = search_semantic_scholar(q, year_from, year_to, max_results=8)
        print(f"      → {len(papers)}편 수집")
        collected.extend(papers)
        time.sleep(0.5)

    # 학회 논문 검색: 의미 있는 토큰 ≥ 2개인 질의를 anchor로 선택
    def _good_anchor_query(candidates: list[str]) -> str:
        for q in candidates:
            toks = [t for t in _valid_query_tokens(q) if _norm_token(t) not in _STOP_WORDS]
            if len(toks) >= 2:
                return q
        return candidates[0] if candidates else ""

    anchor_q = _good_anchor_query(queries)
    for venue in CONF_VENUES:
        venue_query = f"{anchor_q} {venue}"
        print(f"    [S2-venue {year_from}-{year_to}] {venue} (anchor: {anchor_q[:40]})")
        papers = search_semantic_scholar(venue_query, year_from, year_to, max_results=5)
        matched = [p for p in papers if venue.lower() in (p.get("venue") or "").lower()]
        print(f"      → venue 매칭 {len(matched)}편")
        collected.extend(matched)
        time.sleep(0.5)

    return collected


# ──────────────────────────────────────────
# 관련도 재정렬
# ──────────────────────────────────────────

def _build_query_tokens(topic_data: dict) -> tuple[set[str], set[str], set[str]]:
    """강/중/약 가중치 토큰 세트를 반환한다."""
    inp      = topic_data.get("input", {})
    keywords = topic_data.get("search_keywords", {})

    def tok(text: str) -> set[str]:
        return {t.lower() for t in re.split(r"\W+", text or "") if len(t) > 2}

    strong = (tok(inp.get("problem_definition", ""))
              | tok(inp.get("details", ""))
              | tok(inp.get("constraints", ""))
              | tok(inp.get("target_metric", "")))
    medium = set().union(*(tok(kw) for kw in keywords.get("primary", [])))
    weak   = set().union(*(tok(kw) for kw in keywords.get("secondary", [])))
    return strong, medium, weak


def rerank_papers(papers: list[dict], topic_data: dict) -> list[dict]:
    """topic_data 기반 관련도 점수로 논문을 재정렬한다.

    점수 구성:
      - strong tokens (problem/constraints/metric) × 제목 3.0 / 초록 1.0
      - medium tokens (primary keywords)           × 제목 2.0 / 초록 0.7
      - weak tokens   (secondary keywords)         × 제목 1.0 / 초록 0.3
      - sections 존재 가점 0.5
      - 최신성 tie-breaker (0 ~ 0.3)
    """
    strong, medium, weak = _build_query_tokens(topic_data)

    def score(p: dict) -> float:
        def tok(text: str) -> set[str]:
            return {t.lower() for t in re.split(r"\W+", text or "") if len(t) > 2}
        title_tok = tok(p.get("title", ""))
        abs_tok   = tok(p.get("abstract", ""))

        s  = len(strong & title_tok) * 3.0 + len(strong & abs_tok) * 1.0
        s += len(medium & title_tok) * 2.0 + len(medium & abs_tok) * 0.7
        s += len(weak   & title_tok) * 1.0 + len(weak   & abs_tok) * 0.3
        if p.get("sections"):
            s += 0.5
        year = int((p.get("year") or "0")[:4] or "0")
        s += min(max(year - 2015, 0) / 40, 0.3)
        return s

    return sorted(papers, key=score, reverse=True)


# ──────────────────────────────────────────
# 메인 검색 함수
# ──────────────────────────────────────────

def research_papers(topic_file: str) -> dict:
    """
    최신 2년치부터 검색을 시작하고, MIN_PAPERS에 미달하면
    1년씩 소급하여 최대 MAX_YEARS_BACK년까지 확장한다.

    흐름:
      initial retrieval → dedup → rerank
      → rerank 상위 5편 seed → citation expansion + foundational backtracking
      → expanded set rerank → GitHub 추출 → fulltext → 저장
    """
    topic_data = json.loads(Path(topic_file).read_text(encoding="utf-8"))
    keywords   = topic_data.get("search_keywords", {})
    primary    = keywords.get("primary", [])
    secondary  = keywords.get("secondary", [])
    topic_name = topic_data.get("input", {}).get("topic", "research")
    topic_slug = re.sub(r"\W+", "_", topic_name.lower())[:30]
    queries    = _build_search_queries(topic_data)

    cur_year  = datetime.now().year
    year_to   = cur_year
    year_from = cur_year - (INIT_WINDOW - 1)

    all_papers: list[dict] = []
    seen_titles: set[str] = set()
    search_log: list[str] = []

    def _extend_dedup(batch: list[dict]) -> int:
        added = 0
        for p in batch:
            key = re.sub(r"\s+", " ", p["title"].lower().strip())
            if key not in seen_titles:
                seen_titles.add(key)
                all_papers.append(p)
                added += 1
        return added

    print(f"\n  [논문 검색] 최신 {INIT_WINDOW}년 ({year_from}~{year_to}) 우선 검색...")
    print(f"  [검색 질의] {queries}")
    batch = _search_window(queries, year_from, year_to)
    added = _extend_dedup(batch)
    search_log.append(f"{year_from}-{year_to}: {added}편")
    print(f"  → 중복 제거 후 {len(all_papers)}편")

    years_expanded = 0
    while len(all_papers) < MIN_PAPERS and years_expanded < MAX_YEARS_BACK - INIT_WINDOW:
        year_to   = year_from - 1
        year_from = year_to   - 1
        years_expanded += 1
        print(f"\n  [논문 검색] {len(all_papers)}편 < {MIN_PAPERS}편 → {year_from}년 추가 검색...")
        batch = _search_window(queries, year_from, year_to)
        added = _extend_dedup(batch)
        search_log.append(f"{year_from}-{year_to}: {added}편 추가")
        print(f"  → 누적 {len(all_papers)}편")

    # 1차 rerank → 관련도 상위 5편을 seed로 선정
    if all_papers:
        print(f"\n  [Rerank 1] 관련도 재정렬 후 seed 선정...")
        all_papers[:] = rerank_papers(all_papers, topic_data)
        seed_papers   = all_papers[:5]
        print(f"  → seed: {[p['title'][:40] for p in seed_papers]}")

    # Citation expansion (rerank 상위 seed 기반)
    if all_papers:
        print(f"\n  [Citation Expansion] seed 기반 citation 탐색...")
        citation_batch = expand_citations(seed_papers)
        added_cit = _extend_dedup(citation_batch)
        if added_cit:
            search_log.append(f"citation-expansion: {added_cit}편 추가")
            print(f"  → citation {added_cit}편 추가, 누적 {len(all_papers)}편")

    # Foundational backtracking (rerank 상위 seed 기반, 연도 제한 없음)
    if all_papers:
        print(f"\n  [Foundational Backtrack] seed 기반 공통 인용 논문 추적...")
        foundational_batch = fetch_foundational_papers(seed_papers)
        added_found = _extend_dedup(foundational_batch)
        if added_found:
            search_log.append(f"foundational-backtrack: {added_found}편 추가")
            print(f"  → foundational {added_found}편 추가, 누적 {len(all_papers)}편")

    # 2차 rerank (expansion 포함 전체 재정렬) + GitHub 링크 추출
    print(f"\n  [Rerank 2] 확장된 전체 논문 관련도 재정렬...")
    all_papers[:] = rerank_papers(all_papers, topic_data)
    for p in all_papers:
        if not p.get("github"):
            p["github"] = _extract_github(p.get("abstract", ""))

    # 상위 논문 arXiv 섹션 추출 (arxiv_id가 있는 것만)
    print(f"\n  [Full Text] 상위 {TOP_FULLTEXT}편 arXiv 섹션 추출 중...")
    ft_count = 0
    for p in all_papers[:TOP_FULLTEXT]:
        if p.get("arxiv_id"):
            sections = fetch_arxiv_sections(p["arxiv_id"])
            if sections:
                p["sections"] = sections
                ft_count += 1
                print(f"    → {p['title'][:50]}... ({list(sections.keys())})")
            time.sleep(1)
    print(f"  → {ft_count}편 섹션 추출 완료 (총 {len(all_papers)}편 중)")

    result = {
        "timestamp":      datetime.now().isoformat(),
        "topic":          topic_name,
        "query_keywords": queries,
        "year_range":     f"{year_from}~{cur_year}",
        "search_log":     search_log,
        "total_found":    len(all_papers),
        "papers":         all_papers,
    }

    from lab.config import reports_dir
    output_path = reports_dir(topic_slug) / "papers.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  최종 논문 {len(all_papers)}편 저장: {output_path}")
    print(f"  검색 이력: {' | '.join(search_log)}")

    return result


def print_papers(result: dict) -> None:
    papers = result.get("papers", [])
    print(f"\n{'='*60}")
    print(f"  논문 조사 결과: {result['topic']} ({len(papers)}편)")
    print(f"  검색 범위: {result.get('year_range', '')}")
    print(f"{'='*60}")
    for i, p in enumerate(papers[:15], 1):
        print(f"\n[{i}] {p['title']}")
        print(f"    저자: {', '.join(p['authors'][:3])}")
        print(f"    연도: {p['year']}  출처: {p.get('source','')}")
        if p.get("venue"):
            print(f"    학회: {p['venue']}")
        if p.get("github"):
            print(f"    GitHub: {p['github']}")
        print(f"    요약: {p['abstract'][:120]}...")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="논문 조사 (연도 점진 확장)")
    parser.add_argument("--topic-file", required=True, help="주제 분석 JSON 파일")
    args = parser.parse_args()

    result = research_papers(args.topic_file)
    print_papers(result)
