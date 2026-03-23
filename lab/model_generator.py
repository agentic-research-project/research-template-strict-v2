"""
Stage 7: PyTorch 모델 코드 생성

가설과 코드 분석 결과를 기반으로 Claude가 완전한 PyTorch 모델 코드를 생성한다.
생성 코드는 experiments/{topic}_{version}.py 에 저장된다.

사용법:
  python -m lab.model_generator \
    --topic-file      reports/topic_analysis.json \
    --hypothesis-file reports/hypothesis_{topic}.json \
    --code-file       reports/code_analysis_{topic}.json
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from lab.config import query_claude, parse_json


def generate_model(
    topic_file: str,
    hypothesis_file: str,
    code_analysis_file: str,
    version: str = "v1",
) -> dict:
    """
    가설과 코드 분석을 기반으로 PyTorch 모델 코드를 생성한다.

    Returns:
        {"model_file": str, "code": str, "description": str}
    """
    topic = json.loads(Path(topic_file).read_text(encoding="utf-8"))
    hypothesis = json.loads(Path(hypothesis_file).read_text(encoding="utf-8"))
    code_analysis = json.loads(Path(code_analysis_file).read_text(encoding="utf-8"))

    topic_name = topic.get("input", {}).get("topic", "research")
    topic_slug = re.sub(r"\W+", "_", topic_name.lower())[:30]
    inp = topic.get("input", {})
    hyp = hypothesis.get("hypothesis", {})
    exp_plan = hypothesis.get("experiment_plan", {})
    components = code_analysis.get("reusable_components", [])
    tips = code_analysis.get("implementation_tips", [])
    baseline = code_analysis.get("recommended_baseline", "")

    # 컴포넌트 스니펫 요약
    component_info = "\n".join([
        f"- [{c['type']}] {c['name']}: {c['description']}\n  수정사항: {c.get('adaptation_needed', '')}"
        for c in components[:5]
    ])

    # evaluation_metrics 에서 동적으로 expected_metrics 스키마 생성
    metrics_list = exp_plan.get("evaluation_metrics", [])
    metrics_keys = [
        m.split("(")[0].split("—")[0].strip().lower().replace(" ", "_")
        for m in metrics_list[:4]
    ] or ["primary_metric", "secondary_metric"]
    expected_metrics_schema = {k: f"예상 {k} 값" for k in metrics_keys}
    expected_metrics_schema["training_time"] = "예상 학습 시간"

    prompt = f"""당신은 딥러닝 연구 전문가이자 PyTorch 코드 전문가입니다.
아래 연구 가설과 분석 결과를 바탕으로 실행 가능한 완전한 PyTorch 코드를 생성해주세요.

## 연구 주제
- 주제: {inp.get('topic', '')}
- 문제: {inp.get('problem_definition', '')}
- 목표: {inp.get('desired_outcome', '')}
- 제약: {inp.get('constraints', '')}
- 목표 지표: {inp.get('target_metric', '')}

## 연구 가설
{hyp.get('statement_kr', hyp.get('statement', ''))}

## 제안 아키텍처
{exp_plan.get('architecture', '')}

## 데이터셋
{exp_plan.get('dataset', '')}

## 평가 지표
{', '.join(metrics_list)}

## 참고 컴포넌트
{component_info if component_info else '없음 (직접 구현)'}

## 추천 베이스라인
{baseline}

## 구현 팁
{chr(10).join(f'- {t}' for t in tips)}

## 요구사항
1. **완전히 실행 가능한** 단일 Python 파일 생성
2. 다음을 모두 포함:
   - Dataset 클래스 (합성 데이터 또는 실제 데이터 로더)
   - Model 아키텍처 (PyTorch nn.Module)
   - Loss 함수
   - Trainer 클래스 (train/validate 메서드)
   - 메인 실행 블록 (argparse 포함)
3. 모델 출력 형식: stdout에 `METRICS:{{...}}` 줄 출력 (마지막 epoch)
   - METRICS 키는 반드시 위 평가 지표({', '.join(metrics_keys)})를 사용할 것
4. 경량 모델 우선 (빠른 실험을 위해)
5. 합성 데이터 생성 포함 (실제 데이터 없이도 실행 가능)
6. 에러 처리 포함

## 출력 형식
반드시 아래 JSON으로만 답변 (code 필드에 전체 Python 코드):
{{
  "model_name": "모델 이름",
  "description": "모델 설명 (2-3문장)",
  "architecture_summary": "아키텍처 요약",
  "code": "전체 Python 코드 (주석 포함)",
  "requirements": ["torch", "필요한 추가 패키지"],
  "expected_metrics": {json.dumps(expected_metrics_schema, ensure_ascii=False)},
  "usage": "python experiments/{topic_slug}_{version}.py --epochs 10 --batch-size 16"
}}"""

    print("  [Claude SDK] 모델 코드 생성 중...")
    result = parse_json(query_claude(prompt))

    # 코드 파일 저장
    code = result.get("code", "")
    model_file = Path(f"experiments/{topic_slug}_{version}.py")
    model_file.parent.mkdir(exist_ok=True)
    model_file.write_text(code, encoding="utf-8")
    print(f"  모델 저장: {model_file}")

    result["model_file"] = str(model_file)
    result["timestamp"] = datetime.now().isoformat()
    result["topic"] = topic_name
    result["version"] = version

    # 메타 JSON 저장 (코드 제외)
    meta = {k: v for k, v in result.items() if k != "code"}
    meta_path = Path(topic_file).parent / f"model_meta_{version}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  메타 저장: {meta_path}")

    return result


def print_result(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  모델 생성 완료: {result.get('model_name', '')}")
    print(f"{'='*60}")
    print(f"  설명: {result.get('description', '')}")
    print(f"  아키텍처: {result.get('architecture_summary', '')}")
    print(f"  파일: {result.get('model_file', '')}")
    exp = result.get("expected_metrics", {})
    if exp:
        print(f"\n  예상 성능:")
        for k, v in exp.items():
            print(f"    {k}: {v}")
    print(f"\n  실행 방법: {result.get('usage', '')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch 모델 코드 생성")
    parser.add_argument("--topic-file",      required=True)
    parser.add_argument("--hypothesis-file", required=True)
    parser.add_argument("--code-file",       required=True)
    parser.add_argument("--version",         default="v1", help="버전 태그 (기본: v1)")
    args = parser.parse_args()

    result = generate_model(
        args.topic_file,
        args.hypothesis_file,
        args.code_file,
        args.version,
    )
    print_result(result)
