"""
Stage 1: 연구 주제 구조화

사용자 입력과 선택적 참조 이미지를 받아 논문 검색 키워드, 연구 범위,
성공 기준 등 구조화된 연구 질문으로 변환한다.

이미지 활용 예시:
  - 입력 이미지  : 현재 문제를 보여주는 샘플
  - 목표 이미지  : 원하는 출력 품질 예시 (before/after 비교)
  - 실패 사례    : 기존 방법의 한계를 보여주는 결과 이미지

이미지를 제공하면 Claude가 시각적으로 문제를 파악하여
더 정확한 연구 질문과 키워드를 도출한다.

사용법:
  # 텍스트만
  python -m lab.topic_analyzer

  # 이미지 포함
  python -m lab.topic_analyzer \
    --image input_noisy.tif target_clean.jpg reference.bmp \
    --image-labels "노이즈 입력" "목표 출력" "참고 샘플"

지원 형식: jpg, jpeg, png, gif, webp (직접 전송)
           tif, tiff, bmp            (PNG 자동 변환, Pillow 필요)
"""

import argparse
import json
import re
from pathlib import Path

from lab.config import query_claude, parse_json


# ──────────────────────────────────────────
# 키워드 후처리
# ──────────────────────────────────────────

# 단독으로는 검색 효과가 없는 너무 넓은 단어
_GENERIC_TERMS = {
    "classification", "segmentation", "enhancement", "generation",
    "diffusion", "computer", "vision", "deep", "learning", "neural",
    "network", "model", "method", "approach", "image", "images",
    "data", "detection", "recognition",
}


MIN_PRIMARY = 2   # 필터 후 primary 최소 개수


def _tokenize_non_generic(text: str) -> list[str]:
    """텍스트에서 generic term을 제외한 토큰 목록을 반환한다."""
    return [
        t.lower() for t in re.split(r"\W+", text or "")
        if len(t) > 3 and t.lower() not in _GENERIC_TERMS
    ]


def _filter_search_keywords(keywords: dict, topic_data: dict) -> dict:
    """generic term 단독 항목 제거 → secondary 승격 → fallback phrase 생성으로 primary를 MIN_PRIMARY 이상 유지한다."""
    inp = topic_data.get("input", {})

    def is_generic(kw: str) -> bool:
        tokens = {t.lower() for t in re.split(r"\W+", kw) if t}
        return bool(tokens) and tokens.issubset(_GENERIC_TERMS)

    def make_phrase(*fields: str, per_field: int = 2, max_total: int = 4) -> str:
        """필드별 quota를 두고 non-generic 토큰을 합쳐 phrase를 만든다."""
        merged = []
        for f in fields:
            merged.extend(_tokenize_non_generic(f)[:per_field])
        return " ".join(list(dict.fromkeys(merged))[:max_total])

    primary   = [kw for kw in keywords.get("primary",   []) if not is_generic(kw)]
    secondary = [kw for kw in keywords.get("secondary", []) if not is_generic(kw)]
    promoted  = []  # secondary → primary 승격 추적

    # ── 1단계: primary가 부족하면 secondary에서 승격 ──────────────────
    # 토큰 교집합 기반 relevance (substring 오탐 방지)
    _cons_tokens = set(_tokenize_non_generic(inp.get("constraints", "")))
    _met_tokens  = set(_tokenize_non_generic(inp.get("target_metric", "")))
    _det_tokens  = set(_tokenize_non_generic(
        inp.get("details", "") + " " + inp.get("problem_definition", "")
    ))

    def relevance(kw: str) -> int:
        kw_tokens = set(_tokenize_non_generic(kw))
        if kw_tokens & _cons_tokens:  return 3
        if kw_tokens & _met_tokens:   return 2
        if kw_tokens & _det_tokens:   return 1
        return 0

    if len(primary) < MIN_PRIMARY:
        candidates = sorted(secondary, key=relevance, reverse=True)
        for kw in candidates:
            if len(primary) >= MIN_PRIMARY:
                break
            if kw not in primary:
                primary.append(kw)
                promoted.append(kw)

    # ── 2단계: 여전히 부족하면 topic_data["input"]에서 fallback phrase ─
    must_have = _cons_tokens | _met_tokens  # fallback에 반드시 포함돼야 할 토큰
    fallback_sources = [
        (inp.get("problem_definition", ""), inp.get("constraints", "")),
        (inp.get("details", ""),            inp.get("target_metric", "")),
        (inp.get("topic", ""),              inp.get("constraints", "")),
    ]
    for fields in fallback_sources:
        if len(primary) >= MIN_PRIMARY:
            break
        phrase = make_phrase(*fields)
        if not phrase or is_generic(phrase) or phrase in primary:
            continue
        # fallback phrase에 must_have 토큰이 없으면 보강
        if must_have and not (set(_tokenize_non_generic(phrase)) & must_have):
            extra = next(iter(must_have), "")
            if extra:
                phrase = f"{phrase} {extra}".strip()
        primary.append(phrase)

    # ── 3단계: secondary에서 primary로 승격된 항목 제거 (중복 축소) ────
    secondary = [kw for kw in secondary if kw not in promoted]

    # ── 4단계: constraint/metric이 전체 키워드에 없으면 secondary 보강 ─
    existing = " ".join(primary + secondary).lower()
    for field in [inp.get("constraints", ""), inp.get("target_metric", "")]:
        if not field:
            continue
        tokens = [t for t in re.split(r"\W+", field)
                  if len(t) > 3 and t.lower() not in _GENERIC_TERMS
                  and t.lower() not in existing]
        if tokens and len(secondary) < 6:
            secondary.append(" ".join(tokens[:3]))
            existing += " " + " ".join(tokens[:3])

    # ── 최종: 중복 제거 + 2~5개 범위 정리 ───────────────────────────
    primary   = list(dict.fromkeys(primary))[:5]
    secondary = list(dict.fromkeys(secondary))

    if promoted:
        print(f"  [Keyword] secondary→primary 승격: {promoted}")
    print(f"  [Keyword] primary({len(primary)}): {primary}")
    print(f"  [Keyword] secondary({len(secondary)}): {secondary}")

    return {**keywords, "primary": primary, "secondary": secondary}


# ──────────────────────────────────────────
# 핵심 분석 함수
# ──────────────────────────────────────────

def analyze_topic(
    topic: str,
    details: str,
    problem_definition: str,
    desired_outcome: str,
    constraints: str = "",
    target_metric: str = "",
    image_paths: list[str] | None = None,
    image_labels: list[str] | None = None,
) -> dict:
    """
    사용자 입력을 구조화된 연구 질문으로 변환한다.

    Args:
        topic             : 연구 주제
        details           : 구체적 내용
        problem_definition: 문제 정의
        desired_outcome   : 원하는 결과
        constraints       : 제약 조건
        target_metric     : 목표 성능 지표
        image_paths       : 참조 이미지 경로 목록 (선택)
        image_labels      : 각 이미지의 레이블 (선택, 예: ["노이즈 입력", "목표 출력"])

    Returns:
        구조화된 연구 질문 dict
    """
    image_paths  = image_paths  or []
    image_labels = image_labels or []
    has_images   = bool(image_paths)

    # ── 텍스트 프롬프트 ──
    image_instruction = ""
    if has_images:
        labeled = "\n".join(
            f"  - {p} ({image_labels[i] if i < len(image_labels) else f'이미지 {i+1}'})"
            for i, p in enumerate(image_paths)
        )
        image_instruction = f"""
## 이미지 분석 요청
아래 이미지 파일들을 읽어 분석하세요:
{labeled}
분석 항목:
- 이미지에서 관찰되는 문제/현상의 특성 (종류, 분포, 심각도)
- 입력과 목표 출력 간 시각적 차이
- 기존 방법의 실패 패턴 (실패 사례 이미지가 있는 경우)
- 이미지 도메인 특성 (의료/산업/자연/텍스트 등)
이 시각적 분석을 research_question, search_keywords, success_criteria에 반영하세요.
"""

    text_prompt = f"""당신은 딥러닝 연구 전문가입니다.
아래 연구 입력{"과 참조 이미지" if has_images else ""}을 분석하여
체계적인 연구 질문과 논문 검색 전략을 수립해주세요.
{image_instruction}
## 텍스트 입력
- 연구 주제: {topic}
- 구체적 내용: {details}
- 문제 정의: {problem_definition}
- 원하는 결과: {desired_outcome}
- 제약 조건: {constraints if constraints else "없음"}
- 목표 지표: {target_metric if target_metric else "미정"}

## 출력 형식 (반드시 아래 JSON 형식으로만 답변)
{{
  "research_question": "핵심 연구 질문 (1문장, 영어)",
  "background": "연구 배경 및 중요성 (2-3문장)",
  "image_analysis": {{
    "input_characteristics": "입력 이미지 특성 분석 (이미지 없으면 null)",
    "target_characteristics": "목표 출력 특성 (이미지 없으면 null)",
    "visual_gap": "입력-출력 간 시각적 차이 요약 (이미지 없으면 null)",
    "domain_type": "이미지 도메인 (industrial/medical/natural/unknown)"
  }},
  "problem_analysis": {{
    "core_problem": "핵심 문제 요약",
    "current_limitations": ["현재 기술의 한계점들"],
    "why_difficult": "왜 어려운 문제인지"
  }},
  "scope": {{
    "in_scope": ["연구 범위 내 항목들"],
    "out_scope": ["연구 범위 외 항목들"]
  }},
  "search_keywords": {{
    "primary": [
      "핵심 키워드 3-5개 (영어, 반드시 phrase 형태)",
      "규칙: task + domain/object + distinguishing constraint 조합",
      "예시: 'single-scan CT denoising defect-preserving', 'low-dose X-ray reconstruction lightweight'",
      "금지: 'deep learning', 'neural network', 'image enhancement' 같은 단독 generic 단어",
      "요건: constraints와 target_metric의 핵심 표현이 최소 1개 이상 반영될 것"
    ],
    "secondary": [
      "보조 키워드 3-5개 (영어, phrase 형태)",
      "규칙: dataset / metric / failure mode / baseline family / deployment constraint 중심",
      "예시: 'PSNR SSIM image quality', 'self-supervised denoising unpaired', 'real-time inference edge deployment'"
    ],
    "venues": ["NeurIPS", "CVPR", "ECCV", "ICLR", "ICML", "MICCAI", "AAAI"]
  }},
  "success_criteria": {{
    "quantitative": ["수치 목표들 (예: Accuracy >= 95%, F1 >= 0.9, PSNR >= 30dB 등 도메인에 맞게)"],
    "qualitative": ["정성적 목표들"]
  }},
  "constraints": ["제약 조건 목록"],
  "target_metrics": ["평가 지표 목록"],
  "expected_challenges": ["예상되는 연구 도전과제 2-3개"]
}}"""

    if image_paths:
        print(f"  [이미지 {len(image_paths)}장 포함] 시각 분석 활성화")

    print("  [Claude SDK] 주제 분석 중...")
    result = parse_json(query_claude(text_prompt, image_paths=image_paths or None))

    # 원본 입력 보존 (후처리보다 먼저 세팅해야 _filter_search_keywords에서 참조 가능)
    result["input"] = {
        "topic":              topic,
        "details":            details,
        "problem_definition": problem_definition,
        "desired_outcome":    desired_outcome,
        "constraints":        constraints,
        "target_metric":      target_metric,
    }

    # search_keywords 후처리: generic term 제거 + secondary 승격 + fallback 보강
    if "search_keywords" in result:
        result["search_keywords"] = _filter_search_keywords(
            result["search_keywords"], result
        )
        n_primary = len(result["search_keywords"].get("primary", []))
        if n_primary < MIN_PRIMARY:
            print(f"  ⚠️ [Keyword] 필터 후 primary {n_primary}개 — MIN_PRIMARY({MIN_PRIMARY}) 미달")

    if image_paths:
        result["input"]["image_paths"]  = image_paths
        result["input"]["image_labels"] = image_labels

    return result


# ──────────────────────────────────────────
# 출력
# ──────────────────────────────────────────

def print_analysis(analysis: dict) -> None:
    print("\n" + "=" * 60)
    print("  연구 주제 분석 결과")
    print("=" * 60)

    print(f"\n[연구 질문]\n  {analysis.get('research_question', '')}")
    print(f"\n[배경]\n  {analysis.get('background', '')}")

    # 이미지 분석 결과 출력
    ia = analysis.get("image_analysis", {})
    if ia.get("input_characteristics"):
        print(f"\n[이미지 분석]")
        print(f"  도메인: {ia.get('domain_type', '')}")
        print(f"  입력 특성: {ia.get('input_characteristics', '')}")
        print(f"  목표 특성: {ia.get('target_characteristics', '')}")
        print(f"  시각적 차이: {ia.get('visual_gap', '')}")

    pa = analysis.get("problem_analysis", {})
    print(f"\n[문제 분석]")
    print(f"  핵심 문제: {pa.get('core_problem', '')}")
    print(f"  왜 어려운가: {pa.get('why_difficult', '')}")
    for lim in pa.get("current_limitations", []):
        print(f"    - {lim}")

    keywords = analysis.get("search_keywords", {})
    print(f"\n[검색 키워드]")
    print(f"  핵심: {', '.join(keywords.get('primary', []))}")
    print(f"  보조: {', '.join(keywords.get('secondary', []))}")

    sc = analysis.get("success_criteria", {})
    print(f"\n[성공 기준]")
    for q in sc.get("quantitative", []):
        print(f"  (수치) {q}")
    for q in sc.get("qualitative", []):
        print(f"  (정성) {q}")

    print(f"\n[예상 도전과제]")
    for c in analysis.get("expected_challenges", []):
        print(f"  - {c}")
    print("=" * 60)


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="연구 주제 분석 (이미지 선택 입력)")
    parser.add_argument("--topic",              required=True,
                        help="연구 주제 (예: 'medical image segmentation')")
    parser.add_argument("--details",            required=True,
                        help="구체적 내용 (예: 'CT 영상에서 종양 자동 분할')")
    parser.add_argument("--problem-definition", required=True,
                        help="문제 정의")
    parser.add_argument("--desired-outcome",    required=True,
                        help="원하는 결과")
    parser.add_argument("--constraints",        default="",
                        help="제약 조건 (선택)")
    parser.add_argument("--target-metric",      default="",
                        help="목표 지표 (선택, 예: 'Dice Score, IoU')")
    parser.add_argument("--image",   nargs="*", dest="image_paths",  default=[],
                        metavar="PATH",  help="참조 이미지 경로 (여러 개 가능)")
    parser.add_argument("--image-labels", nargs="*", default=[],
                        metavar="LABEL", help="각 이미지 레이블 (예: '노이즈 입력' '목표 출력')")
    args = parser.parse_args()

    result = analyze_topic(
        topic=args.topic,
        details=args.details,
        problem_definition=args.problem_definition,
        desired_outcome=args.desired_outcome,
        constraints=args.constraints,
        target_metric=args.target_metric,
        image_paths=args.image_paths,
        image_labels=args.image_labels,
    )
    print_analysis(result)

    topic_slug = re.sub(r"\W+", "_", args.topic.lower())[:30]
    out = Path("reports") / topic_slug / "topic_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  저장: {out}")
