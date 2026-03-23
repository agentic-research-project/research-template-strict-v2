"""
Stage 8: Research Loop - PyTorch 모델 실험 실행

생성된 모델 코드를 실행하고 결과를 results/{experiment_id}.json에 저장한다.

사용법:
  python -m lab.research_loop --model-file models/denoising_v1.py
"""

import argparse
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


def run_experiment(model_file: str, experiment_id: str = None) -> dict:
    """모델 실험을 실행하고 결과를 저장한다."""
    if experiment_id is None:
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model_path = Path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_file}")

    print(f"  실험 시작: {experiment_id}")
    print(f"  모델 파일: {model_file}")

    result = {
        "experiment_id": experiment_id,
        "model_file": model_file,
        "timestamp": datetime.now().isoformat(),
        "status": "running",
        "metrics": {},
        "logs": [],
        "error": None,
    }

    try:
        proc = subprocess.run(
            [sys.executable, str(model_path)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1시간 제한
        )

        result["logs"] = proc.stdout.splitlines()
        result["status"] = "completed" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            result["error"] = proc.stderr[-2000:]  # 마지막 2000자만 저장

        # stdout에서 metrics 파싱 시도 (모델이 JSON 출력 시)
        for line in proc.stdout.splitlines():
            if line.startswith("METRICS:"):
                try:
                    result["metrics"] = json.loads(line[8:])
                except Exception:
                    pass

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "실험 시간 초과 (1시간)"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    # 결과 저장
    output_path = Path(f"results/{experiment_id}.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  결과 저장: {output_path}")
    print(f"  상태: {result['status']}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch 모델 실험 실행")
    parser.add_argument("--model-file", required=True, help="실행할 모델 파이썬 파일")
    parser.add_argument("--experiment-id", default=None, help="실험 ID (기본: 타임스탬프)")
    args = parser.parse_args()

    result = run_experiment(args.model_file, args.experiment_id)
    print(f"\n실험 완료: {result['experiment_id']} - {result['status']}")
