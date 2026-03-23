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

from lab.config import GITHUB_TOKEN


ARXIV_API = "https://export.arxiv.org/api/query"
S2_API    = "https://api.semanticscholar.org/graph/v1"

MIN_PAPERS    = 10   # 이 수 이상이면 연도 확장 중단
MAX_YEARS_BACK = 6   # 최대 소급 연수 (2년 기본 + 최대 4년 추가)
INIT_WINDOW   = 2    # 첫 검색 연도 윈도우 (현재 연도 포함 2년)


def _http_get(url: str, headers: dict = None) -> str:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


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
    if year_from and year_to:
        date_filter = f"submittedDate:[{year_from}0101 TO {year_to}1231]"
        search_q = f"all:{query} AND {date_filter}"
    else:
        search_q = f"all:{query}"

    params = urllib.parse.urlencode({
        "search_query": search_q,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
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

        papers.append({
            "title":     tag("title").replace("\n", " "),
            "authors":   authors[:5],
            "year":      published[:4],
            "published": published,
            "abstract":  tag("summary").replace("\n", " ")[:500],
            "url":       arxiv_id,
            "arxiv_id":  arxiv_id.split("/abs/")[-1] if "/abs/" in arxiv_id else "",
            "github":    "",
            "source":    "arXiv",
        })

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
    fields = "title,authors,year,abstract,externalIds,venue"
    qs: dict = {
        "query":  query,
        "limit":  max_results,
        "fields": fields,
    }
    if year_from and year_to:
        qs["year"] = f"{year_from}-{year_to}"

    url = f"{S2_API}/paper/search?{urllib.parse.urlencode(qs)}"

    try:
        data = json.loads(_http_get(url))
    except Exception as e:
        print(f"    [S2] 요청 실패: {e}")
        return []

    papers = []
    for p in data.get("data", []):
        authors  = [a.get("name", "") for a in p.get("authors", [])[:5]]
        ext      = p.get("externalIds", {})
        arxiv_id = ext.get("ArXiv", "")
        url_link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""

        papers.append({
            "title":     p.get("title", ""),
            "authors":   authors,
            "year":      str(p.get("year", "")),
            "published": str(p.get("year", "")),
            "abstract":  (p.get("abstract") or "")[:500],
            "url":       url_link,
            "arxiv_id":  arxiv_id,
            "github":    "",
            "venue":     p.get("venue", ""),
            "source":    "SemanticScholar",
        })

    return papers


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
# 연도 범위 점진 확장 검색
# ──────────────────────────────────────────

def _search_window(
    primary: list[str],
    secondary: list[str],
    year_from: int,
    year_to: int,
) -> list[dict]:
    """주어진 연도 구간에서 논문을 검색하여 반환한다."""
    collected = []

    for kw in primary[:3]:
        print(f"    [arXiv {year_from}-{year_to}] {kw}")
        papers = search_arxiv(kw, year_from, year_to, max_results=8)
        collected.extend(papers)
        time.sleep(1)

    for kw in (primary + secondary)[:4]:
        print(f"    [S2 {year_from}-{year_to}] {kw}")
        papers = search_semantic_scholar(kw, year_from, year_to, max_results=8)
        collected.extend(papers)
        time.sleep(0.5)

    return collected


# ──────────────────────────────────────────
# 메인 검색 함수
# ──────────────────────────────────────────

def research_papers(topic_file: str) -> dict:
    """
    최신 2년치부터 검색을 시작하고, MIN_PAPERS에 미달하면
    1년씩 소급하여 최대 MAX_YEARS_BACK년까지 확장한다.
    """
    topic_data = json.loads(Path(topic_file).read_text(encoding="utf-8"))
    keywords   = topic_data.get("search_keywords", {})
    primary    = keywords.get("primary", [])
    secondary  = keywords.get("secondary", [])
    topic_name = topic_data.get("input", {}).get("topic", "research")
    topic_slug = re.sub(r"\W+", "_", topic_name.lower())[:30]

    cur_year = datetime.now().year
    year_to  = cur_year
    year_from = cur_year - (INIT_WINDOW - 1)   # 기본: 최근 2년

    all_papers: list[dict] = []
    seen_titles: set[str] = set()
    search_log: list[str] = []

    def _extend_dedup(batch: list[dict]) -> int:
        """배치를 전체 목록에 추가하되 중복은 건너뛴다. 추가된 수를 반환."""
        added = 0
        for p in batch:
            key = re.sub(r"\s+", " ", p["title"].lower().strip())
            if key not in seen_titles:
                seen_titles.add(key)
                all_papers.append(p)
                added += 1
        return added

    print(f"\n  [논문 검색] 최신 {INIT_WINDOW}년 ({year_from}~{year_to}) 우선 검색...")
    batch = _search_window(primary, secondary, year_from, year_to)
    added = _extend_dedup(batch)
    search_log.append(f"{year_from}-{year_to}: {added}편")
    print(f"  → 중복 제거 후 {len(all_papers)}편")

    years_expanded = 0
    while len(all_papers) < MIN_PAPERS and years_expanded < MAX_YEARS_BACK - INIT_WINDOW:
        year_to   = year_from - 1
        year_from = year_to   - 1   # 1년 단위 슬라이딩
        years_expanded += 1
        print(f"\n  [논문 검색] {len(all_papers)}편 < {MIN_PAPERS}편 → {year_from}년 추가 검색...")
        batch = _search_window(primary, secondary, year_from, year_to)
        added = _extend_dedup(batch)
        search_log.append(f"{year_from}-{year_to}: {added}편 추가")
        print(f"  → 누적 {len(all_papers)}편")

    # GitHub 링크 추출 + 최신순 정렬
    for p in all_papers:
        if not p.get("github"):
            p["github"] = _extract_github(p.get("abstract", ""))

    all_papers.sort(key=lambda x: x.get("published", x.get("year", "0")), reverse=True)

    result = {
        "timestamp":      datetime.now().isoformat(),
        "topic":          topic_name,
        "query_keywords": primary + secondary,
        "year_range":     f"{year_from}~{cur_year}",
        "search_log":     search_log,
        "total_found":    len(all_papers),
        "papers":         all_papers,
    }

    output_path = Path("reports") / topic_slug / "papers.json"
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
