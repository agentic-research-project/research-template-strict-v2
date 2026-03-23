"""
Stage 4: 다중 LLM 가설 검증 + 자동 개선 루프

GPT + Gemini로 가설을 검증하고, 다중 기준 미달 시 Claude가 약점을 분석하여
가설을 개선한 뒤 재검증한다.

통과 기준 (3가지 모두 충족 필요):
  1. 평균 점수 ≥ 8.5
  2. 각 항목(novelty/validity/feasibility/impact) 평균 ≥ 8.0
  3. 두 평가자 점수 차이 ≤ 1.5

Novelty 특칙: 선행연구 3~5개와 비교표 필수 → prior_art_comparison 필드

검증 전략 (critique-revise 루프):
  1. GPT + Gemini 비판적 검토 → 약점·제안·선행연구 비교표 수집
  2. Claude가 약점을 구조화된 방식으로 해소 → 가설 재작성
  3. 재검증 → 모든 기준 달성 or 최대 반복 도달 시 종료

사용법:
  # 단순 검증
  python -m lab.hypothesis_validator --hypothesis-file reports/hypothesis.json

  # 자동 개선 루프 (목표 8.5점)
  python -m lab.hypothesis_validator \\
    --hypothesis-file reports/hypothesis.json \\
    --topic-file      reports/topic_analysis.json \\
    --refine --target-score 8.5 --max-iter 3
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from lab.config import (
    OPENAI_MODEL, GEMINI_MODEL,
    query_claude, get_openai_client, get_gemini_model, parse_json,
)

SCORE_THRESHOLD = 8.5   # 기본 목표 점수


# ──────────────────────────────────────────
# 공통 검증 프롬프트
# ──────────────────────────────────────────

def _build_validation_prompt(hyp_content: dict, extra_instruction: str = "") -> str:
    """GPT/Gemini 공통 검증 프롬프트를 반환한다."""
    return f"""당신은 딥러닝 연구 전문가입니다. 아래 연구 가설을 비판적으로 검토해주세요.

## 연구 가설
{json.dumps(hyp_content, ensure_ascii=False, indent=2)}

## 검토 기준
1. 참신성 (Novelty): 기존 연구 대비 새로운 기여가 있는가?
   ⚠️ **Novelty 채점 필수 조건**: 반드시 비슷한 선행연구 3~5개와 이 가설의 차이점을 표 형식으로 비교한 뒤에 점수를 부여하세요.
   비교표 형식 (prior_art_comparison 필드):
   | 선행연구 | 핵심 방법 | 본 가설과의 차이 |
   각 행마다 선행연구명, 핵심방법, 차이점을 명시하세요.

2. 타당성 (Validity): 가설이 논리적으로 일관성 있는가?
3. 실현 가능성 (Feasibility): 현재 기술로 구현 가능한가?
4. 영향력 (Impact): 성공 시 해당 분야에 의미 있는 기여를 하는가?

각 기준을 고려하여 10점 만점으로 채점하세요 (9-10: 탁월, 7-8: 양호, 5-6: 보통, <5: 부족).
통과 기준: 평균 ≥ 8.5, 각 항목 ≥ 8.0.
{extra_instruction}
반드시 아래 JSON 형식으로만 답변하세요 (마크다운 없이):
{{
  "score": 점수(1-10),
  "score_breakdown": {{"novelty": 0, "validity": 0, "feasibility": 0, "impact": 0}},
  "prior_art_comparison": [
    {{"paper": "선행연구명", "method": "핵심 방법", "difference": "본 가설과의 차이"}}
  ],
  "feasibility": "실현 가능성 평가",
  "strengths": ["강점1", "강점2"],
  "weaknesses": ["약점1 (구체적으로)", "약점2 (구체적으로)"],
  "suggestions": ["개선안1 (실행 가능한 수준으로)", "개선안2"],
  "verdict": "approve/revise/reject"
}}"""


# ──────────────────────────────────────────
# 다중 기준 통과 판정
# ──────────────────────────────────────────

def _passes_validation(gpt: dict, gem: dict, target_score: float = SCORE_THRESHOLD) -> bool:
    """
    세 가지 기준을 모두 충족해야 통과:
      1. 평균 점수 ≥ target_score
      2. 각 항목 최저점 ≥ 8.0  (novelty/validity/feasibility/impact 개별 평균)
      3. 평가자 간 점수 차이 ≤ 1.5
    """
    gpt_score = gpt.get("score", 0)
    gem_score = gem.get("score", 0)
    avg = (gpt_score + gem_score) / 2

    # 기준 1: 평균
    if avg < target_score:
        return False

    # 기준 2: 항목별 최저점
    criteria = ["novelty", "validity", "feasibility", "impact"]
    gpt_bd = gpt.get("score_breakdown", {})
    gem_bd = gem.get("score_breakdown", {})
    for c in criteria:
        item_avg = (gpt_bd.get(c, 0) + gem_bd.get(c, 0)) / 2
        if item_avg < 8.0:
            return False

    # 기준 3: 평가자 간 점수 차이
    if abs(gpt_score - gem_score) > 1.5:
        return False

    return True


# ──────────────────────────────────────────
# GPT-4o 검증
# ──────────────────────────────────────────

def validate_with_gpt4o(hypothesis: dict) -> dict:
    hyp_content = hypothesis.get("hypothesis", hypothesis)
    prompt = _build_validation_prompt(hyp_content)
    response = get_openai_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ──────────────────────────────────────────
# Gemini 검증
# ──────────────────────────────────────────

def validate_with_gemini(hypothesis: dict) -> dict:
    hyp_content = hypothesis.get("hypothesis", hypothesis)
    prompt = _build_validation_prompt(hyp_content)
    response = get_gemini_model().generate_content(prompt)
    return parse_json(response.text)


# ──────────────────────────────────────────
# Claude 가설 개선 (critique-revise)
# ──────────────────────────────────────────

def refine_hypothesis_with_claude(
    hypothesis: dict,
    validation: dict,
    topic: dict,
    iteration: int,
) -> dict:
    """
    GPT-4o + Gemini의 피드백을 바탕으로 Claude가 가설을 개선한다.

    개선 전략:
    - 약점마다 구체적인 수정 방향을 도출
    - 기존 강점은 보존하면서 약점만 해소
    - 실현 가능성을 낮추지 않는 범위에서 참신성 강화
    """
    summary = validation.get("summary", {})
    gpt  = validation.get("gpt4o", {})
    gem  = validation.get("gemini", {})
    inp  = topic.get("input", {}) if topic else {}

    all_weaknesses  = list(dict.fromkeys(
        gpt.get("weaknesses", []) + gem.get("weaknesses", [])
    ))
    all_suggestions = list(dict.fromkeys(
        gpt.get("suggestions", []) + gem.get("suggestions", [])
    ))
    gpt_breakdown = gpt.get("score_breakdown", {})
    gem_breakdown = gem.get("score_breakdown", {})

    # 점수 낮은 기준 파악
    criteria = ["novelty", "validity", "feasibility", "impact"]
    avg_breakdown = {
        c: round((gpt_breakdown.get(c, 5) + gem_breakdown.get(c, 5)) / 2, 1)
        for c in criteria
    }
    weak_criteria = [c for c, v in avg_breakdown.items() if v < 8]

    prompt = f"""당신은 딥러닝 연구 전문가입니다.
아래 연구 가설이 LLM 검증에서 {summary.get('average_score', 0)}/10점을 받았습니다.
목표 점수 {SCORE_THRESHOLD}점에 미달하여 가설을 개선해야 합니다.
이번이 {iteration}번째 개선 시도입니다.

## 현재 가설
{json.dumps(hypothesis.get('hypothesis', hypothesis), ensure_ascii=False, indent=2)}

## 연구 맥락
- 주제: {inp.get('topic', '')}
- 문제 정의: {inp.get('problem_definition', '')}
- 원하는 결과: {inp.get('desired_outcome', '')}
- 제약 조건: {inp.get('constraints', '')}

## 점수 분석 (평균)
{json.dumps(avg_breakdown, ensure_ascii=False)}
→ 개선 필요 기준: {', '.join(weak_criteria) if weak_criteria else '전반적 강화 필요'}

## GPT-4o 지적 약점
{json.dumps(all_weaknesses, ensure_ascii=False)}

## 개선 제안 (GPT + Gemini)
{json.dumps(all_suggestions, ensure_ascii=False)}

## 개선 지침
1. 각 약점에 대해 **구체적인 해소 방안**을 가설에 반영하라
2. 기존 강점({', '.join(gpt.get('strengths', [])[:2])})은 반드시 유지하라
3. {', '.join(weak_criteria)} 기준을 집중적으로 강화하라
4. 가설은 여전히 실험 가능한 수준을 유지하라

## 출력 형식 (JSON만)
{{
  "statement": "개선된 가설 (영어 1문장)",
  "statement_kr": "개선된 가설 (한국어 1문장)",
  "rationale": "개선 근거 (기존과 무엇이 달라졌는지 명시)",
  "key_innovation": "핵심 혁신 포인트",
  "expected_mechanism": "예상 작동 원리",
  "improvement_log": "이번 개선에서 수정한 내용 요약"
}}"""

    print(f"    [Claude SDK] 가설 개선 중 (iteration {iteration})...")
    refined = parse_json(query_claude(prompt))

    # 원본 가설 구조 유지하되 hypothesis 필드만 교체
    updated = dict(hypothesis)
    if "hypothesis" in updated:
        updated["hypothesis"].update(refined)
    else:
        updated.update(refined)
    updated["_refinement_log"] = updated.get("_refinement_log", [])
    updated["_refinement_log"].append({
        "iteration": iteration,
        "prev_score": summary.get("average_score", 0),
        "improvement": refined.get("improvement_log", ""),
    })
    return updated


# ──────────────────────────────────────────
# 단순 검증
# ──────────────────────────────────────────

def _score_and_build(hypothesis: dict, target_score: float = SCORE_THRESHOLD) -> dict:
    """GPT + Gemini 병렬 검증 후 validation dict 반환."""
    print(f"    [{OPENAI_MODEL} + {GEMINI_MODEL}] 병렬 검증 중...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_gpt = executor.submit(validate_with_gpt4o, hypothesis)
        fut_gem = executor.submit(validate_with_gemini, hypothesis)
        gpt = fut_gpt.result()
        gem = fut_gem.result()
    print(f"    → GPT: {gpt.get('score')}/10  Gemini: {gem.get('score')}/10")

    gpt_score = gpt.get("score", 0)
    gem_score = gem.get("score", 0)
    avg = round((gpt_score + gem_score) / 2, 1)
    score_diff = round(abs(gpt_score - gem_score), 1)

    # 항목별 평균
    criteria = ["novelty", "validity", "feasibility", "impact"]
    gpt_bd = gpt.get("score_breakdown", {})
    gem_bd = gem.get("score_breakdown", {})
    avg_breakdown = {
        c: round((gpt_bd.get(c, 0) + gem_bd.get(c, 0)) / 2, 1)
        for c in criteria
    }

    passed = _passes_validation(gpt, gem, target_score)
    verdicts = [gpt.get("verdict"), gem.get("verdict")]
    overall = (
        "approve" if passed
        else "reject" if verdicts.count("reject") >= 2
        else "revise"
    )

    print(f"    → 평균: {avg}/10  항목별: {avg_breakdown}  평가자차이: {score_diff}  통과: {passed}")

    return {
        "gpt4o":  gpt,
        "gemini": gem,
        "summary": {
            "average_score":    avg,
            "avg_breakdown":    avg_breakdown,
            "score_diff":       score_diff,
            "passed_criteria":  passed,
            "overall_verdict":  overall,
            "all_strengths":    gpt.get("strengths", []) + gem.get("strengths", []),
            "all_weaknesses":   gpt.get("weaknesses", []) + gem.get("weaknesses", []),
            "all_suggestions":  gpt.get("suggestions", []) + gem.get("suggestions", []),
        },
    }


def validate_hypothesis(hypothesis_file: str, target_score: float = SCORE_THRESHOLD) -> dict:
    """GPT + Gemini 단순 1회 검증 (다중 기준 판정 포함)."""
    hypothesis_path = Path(hypothesis_file)
    if not hypothesis_path.exists():
        raise FileNotFoundError(f"가설 파일 없음: {hypothesis_file}")

    hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    validation = _score_and_build(hypothesis, target_score)
    validation["timestamp"] = datetime.now().isoformat()
    validation["hypothesis_file"] = hypothesis_file

    reports_dir = Path(hypothesis_file).parent
    output_path = reports_dir / "validation.json"
    output_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    결과 저장: {output_path}")
    return validation


# ──────────────────────────────────────────
# 자동 개선 루프
# ──────────────────────────────────────────

def refine_and_validate(
    hypothesis_file: str,
    topic_file: str = "",
    target_score: float = SCORE_THRESHOLD,
    max_iterations: int = 3,
) -> dict:
    """
    목표 점수에 달할 때까지 critique-revise 루프를 반복한다.

    흐름:
      검증 → 점수 확인 → (미달) Claude 개선 → 재검증 → ...
                       → (달성) 완료

    Returns:
        최종 validation dict (iteration_history 포함)
    """
    hypothesis_path = Path(hypothesis_file)
    topic_path = Path(topic_file) if topic_file else None

    hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    topic = json.loads(topic_path.read_text(encoding="utf-8")) if (topic_path and topic_path.exists()) else {}

    # reports_dir: hypothesis.json 의 부모 디렉터리 (= reports/{slug}/)
    reports_dir = hypothesis_path.parent

    history = []
    best_validation = None
    best_score = 0.0
    current_hypothesis = hypothesis

    print(f"\n  [검증 루프] 목표 점수: {target_score}/10  최대 {max_iterations}회")
    print("=" * 60)

    for i in range(1, max_iterations + 1):
        print(f"\n  ── Iteration {i}/{max_iterations} ──")
        validation = _score_and_build(current_hypothesis, target_score)
        avg = validation["summary"]["average_score"]
        passed = validation["summary"]["passed_criteria"]
        avg_bd = validation["summary"]["avg_breakdown"]
        diff = validation["summary"]["score_diff"]
        print(f"  → 평균: {avg}/10  통과: {passed}  (항목최저: {min(avg_bd.values()):.1f}, 평가자차이: {diff})")

        history.append({
            "iteration": i,
            "score":     avg,
            "passed":    passed,
            "avg_breakdown": avg_bd,
            "score_diff": diff,
            "hypothesis_snapshot": current_hypothesis.get("hypothesis", {}).get("statement_kr", ""),
        })

        if avg > best_score:
            best_score = avg
            best_validation = validation
            best_validation["hypothesis"] = current_hypothesis  # 최고 점수 가설 보존

        if passed:
            print(f"\n  모든 검증 기준 통과! (평균 {avg} ≥ {target_score}, 항목최저 {min(avg_bd.values()):.1f} ≥ 8.0, 평가자차이 {diff} ≤ 1.5)")
            break

        # 미통과 이유 출력
        reasons = []
        if avg < target_score:
            reasons.append(f"평균 {avg} < {target_score}")
        low_items = [c for c, v in avg_bd.items() if v < 8.0]
        if low_items:
            reasons.append(f"항목 미달: {', '.join(low_items)}")
        if diff > 1.5:
            reasons.append(f"평가자 점수차 {diff} > 1.5")
        print(f"  미통과 이유: {' / '.join(reasons)}")

        if i < max_iterations:
            print(f"  → Claude 가설 개선 시작...")
            current_hypothesis = refine_hypothesis_with_claude(
                current_hypothesis, validation, topic, i
            )
            # 개선된 가설을 파일에 저장
            refined_path = reports_dir / f"hypothesis_refined_iter{i}.json"
            refined_path.write_text(
                json.dumps(current_hypothesis, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"  개선 가설 저장: {refined_path}")
        else:
            print(f"\n  최대 반복 도달 ({max_iterations}회). 최고 점수: {best_score}/10")

    # 최종 결과 저장
    goal_achieved = best_validation["summary"].get("passed_criteria", False)
    best_validation["timestamp"] = datetime.now().isoformat()
    best_validation["hypothesis_file"] = hypothesis_file
    best_validation["refinement_summary"] = {
        "iterations_run":  len(history),
        "target_score":    target_score,
        "achieved_score":  best_score,
        "goal_achieved":   goal_achieved,
        "iteration_history": history,
    }

    # 최종 validation 저장
    output_val = reports_dir / "validation.json"
    output_val.write_text(json.dumps(best_validation, ensure_ascii=False, indent=2), encoding="utf-8")

    if goal_achieved and len(history) > 1:
        # 최종 개선된 가설 덮어쓰기
        hypothesis_path.write_text(
            json.dumps(best_validation["hypothesis"], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  최종 개선 가설 저장: {hypothesis_file}")

    print(f"\n  검증 완료  |  최종 점수: {best_score}/10  |  목표 달성: {goal_achieved}")
    return best_validation


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다중 LLM 가설 검증 + 자동 개선")
    parser.add_argument("--hypothesis-file", required=True)
    parser.add_argument("--topic-file",      default="", help="개선 루프에 필요한 주제 파일")
    parser.add_argument("--refine",          action="store_true", help="자동 개선 루프 활성화")
    parser.add_argument("--target-score",    type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--max-iter",        type=int,   default=3)
    args = parser.parse_args()

    if args.refine:
        result = refine_and_validate(
            args.hypothesis_file,
            args.topic_file,
            args.target_score,
            args.max_iter,
        )
        rs = result["refinement_summary"]
        print(f"\n최종 점수: {rs['achieved_score']}/10  |  목표 달성: {rs['goal_achieved']}")
        for h in rs["iteration_history"]:
            print(f"  Iter {h['iteration']}: {h['score']}/10")
    else:
        result = validate_hypothesis(args.hypothesis_file)
        print(f"\n종합 평점: {result['summary']['average_score']}/10")
        print(f"종합 판정: {result['summary']['overall_verdict']}")
