"""
Stage 6: GitHub 코드 분석

GitHub API로 관련 레포를 검색하고 주요 파일을 분석하여
재사용 가능한 컴포넌트와 코드 스니펫을 추출한다.
결과는 reports/code_analysis_{topic}.json에 저장한다.

사용법:
  python -m lab.code_analyzer \
    --topic-file   reports/topic_analysis.json \
    --hypothesis-file reports/hypothesis_{topic}.json
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

from lab.config import GITHUB_TOKEN, query_claude, parse_json


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
# GitHub 레포 검색
# ──────────────────────────────────────────

def search_repos(query: str, max_results: int = 5) -> list[dict]:
    """키워드로 GitHub 레포를 검색한다."""
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
            "full_name": r["full_name"],
            "description": r.get("description", ""),
            "stars": r.get("stargazers_count", 0),
            "url": r.get("html_url", ""),
            "topics": r.get("topics", []),
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
                # 1단계 서브디렉토리만 탐색
                sub = get_repo_structure(full_name, item["path"])
                files.extend(sub[:10])  # 서브dir당 최대 10개
    return files


_GENERIC_FILE_KW = ("model", "train", "network", "arch", "loss", "dataset", "utils")

def _is_relevant_file(path: str, topic_keywords: tuple[str, ...] = ()) -> bool:
    """모델/학습 관련 파일인지 판단 (범용 키워드 + 주제 키워드 결합)."""
    all_kw = _GENERIC_FILE_KW + tuple(kw.lower() for kw in topic_keywords)
    name = path.lower()
    return any(kw in name for kw in all_kw) and name.endswith(".py")


def get_file_content(full_name: str, file_path: str) -> str:
    """파일 내용을 가져온다 (최대 200줄)."""
    data = _gh_get(f"/repos/{full_name}/contents/{file_path}")
    if isinstance(data, dict) and data.get("encoding") == "base64":
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        lines = content.splitlines()[:200]
        return "\n".join(lines)
    return ""


# ──────────────────────────────────────────
# Claude로 코드 분석
# ──────────────────────────────────────────

def _analyze_code_with_claude(
    hypothesis: dict,
    repo_codes: list[dict],
) -> dict:
    """Claude로 수집된 코드를 분석하여 재사용 컴포넌트를 추출한다."""
    hyp = hypothesis.get("hypothesis", {})
    exp_plan = hypothesis.get("experiment_plan", {})

    code_snippets = "\n\n".join([
        f"### {c['repo']} / {c['file']}\n```python\n{c['content'][:600]}\n```"
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
        분석 결과 dict (reports/code_analysis_{topic}.json에도 저장)
    """
    topic = json.loads(Path(topic_file).read_text(encoding="utf-8"))
    hypothesis = json.loads(Path(hypothesis_file).read_text(encoding="utf-8"))

    topic_name = topic.get("input", {}).get("topic", "research")
    topic_slug = re.sub(r"\W+", "_", topic_name.lower())[:30]
    keywords = topic.get("search_keywords", {}).get("primary", [])

    # 파일 필터 키워드: topic_analysis 의 primary + secondary 키워드 단어 추출
    secondary_kw = topic.get("search_keywords", {}).get("secondary", [])
    topic_file_kw = tuple(
        word.lower()
        for kw in (keywords + secondary_kw)[:8]
        for word in kw.split()
        if len(word) > 3
    )

    # 레포 검색
    repos = []
    for kw in keywords[:3]:
        print(f"    [GitHub] 검색: {kw}")
        found = search_repos(kw, max_results=4)
        repos.extend(found)
        time.sleep(1)

    # 중복 제거 (full_name 기준)
    seen = set()
    unique_repos = []
    for r in repos:
        if r["full_name"] not in seen:
            seen.add(r["full_name"])
            unique_repos.append(r)

    print(f"    레포 {len(unique_repos)}개 발견")

    # 관련 파일 수집
    repo_codes = []
    for repo in unique_repos[:4]:
        fname = repo["full_name"]
        print(f"    [GitHub] 파일 분석: {fname}")
        files = get_repo_structure(fname)
        relevant = [f for f in files if _is_relevant_file(f, topic_file_kw)][:3]
        for fpath in relevant:
            content = get_file_content(fname, fpath)
            if content:
                repo_codes.append({
                    "repo": fname,
                    "file": fpath,
                    "content": content,
                })
        time.sleep(0.5)

    print(f"    코드 파일 {len(repo_codes)}개 수집")

    # Claude로 분석
    analysis = {}
    if repo_codes:
        print("    [Claude] 코드 분석 중...")
        analysis = _analyze_code_with_claude(hypothesis, repo_codes)
    else:
        analysis = {
            "reusable_components": [],
            "architecture_insights": [],
            "recommended_baseline": "코드를 찾지 못했습니다. 직접 구현이 필요합니다.",
            "implementation_tips": [],
        }

    result = {
        "timestamp": datetime.now().isoformat(),
        "topic": topic_name,
        "repos_found": [r["full_name"] for r in unique_repos],
        "files_analyzed": len(repo_codes),
        **analysis,
    }

    output_path = Path(topic_file).parent / "code_analysis.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    코드 분석 저장: {output_path}")

    return result


def print_analysis(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  코드 분석 결과: {result['topic']}")
    print(f"{'='*60}")
    print(f"\n발견 레포: {', '.join(result.get('repos_found', []))}")
    print(f"분석 파일 수: {result.get('files_analyzed', 0)}")

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
