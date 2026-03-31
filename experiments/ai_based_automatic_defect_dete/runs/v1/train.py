"""
train.py — PyTorch Fabric 학습 진입점 (unsupervised anomaly detection)

규칙:
- Fabric은 이 파일에서만 초기화
- 마지막 stdout 줄은 반드시: METRICS:{...json...}
- 모든 하이퍼파라미터는 configs/default.yaml에서 로드
- 이 모델은 training-free (backbone frozen): "학습" = reference bank 구축
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
import yaml
from lightning.fabric import Fabric

from data import build_dataloaders
from model import build_model
from module import TrainingModule


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config: dict) -> dict:
    """Reference bank 구축 + 이상탐지 검증."""
    fabric = Fabric(
        accelerator=config.get("accelerator", "auto"),
        precision=config.get("precision", "32-true"),
        devices=config.get("devices", 1),
        strategy=config.get("strategy", "auto"),
    )
    fabric.launch()
    fabric.seed_everything(config["seed"])

    # 모델 생성 (모든 파라미터 frozen — gradient 불필요)
    model = build_model(config)
    # Fabric에 모델 setup 없이 수동으로 device 이동
    model = model.to(fabric.device)
    model.eval()

    # DataLoader 생성 + Fabric wrap
    train_loader, val_loader = build_dataloaders(config)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # 아티팩트 디렉토리
    ckpt_dir = Path("artifacts/checkpoints")
    met_dir  = Path("artifacts/metrics")
    for d in [ckpt_dir, met_dir]:
        d.mkdir(parents=True, exist_ok=True)

    module = TrainingModule(model, config)

    best_val_metric = float("-inf")
    epoch_log_path  = met_dir / "per_epoch_metrics.jsonl"
    pm = config.get("primary_metric", "auroc")

    # 학습 루프 (reference bank 구축 + threshold 보정)
    val_metrics = {}
    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        # reference bank 구축 (train_epoch은 normal patches로 FAISS 인덱스 구축)
        train_metrics = module.train_epoch(train_loader, epoch)

        # 이상탐지 검증
        with torch.no_grad():
            val_metrics = module.val_epoch(val_loader, epoch)

        epoch_row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
            "elapsed_s": round(time.time() - t0, 1),
        }
        with open(epoch_log_path, "a") as fh:
            fh.write(json.dumps(epoch_row) + "\n")

        primary = float(val_metrics.get(pm, 0.0))
        fabric.print(
            f"Epoch {epoch:3d}/{config['epochs']} "
            f"| train_patches={train_metrics.get('num_patches', 0)} "
            f"| val_{pm}={primary:.4f}"
        )

        # 체크포인트 저장 (frozen model이므로 state_dict 저장)
        ckpt = {
            "epoch": epoch,
            "model_state": {k: v for k, v in model.state_dict().items()
                            if not k.startswith("feature_extractor")},
            "evt_thresholds": model.evt_thresholds,
            "faiss_indices": None,  # FAISS index는 별도 저장
            "config": config,
        }
        torch.save(ckpt, str(ckpt_dir / "last.ckpt"))
        if primary > best_val_metric:
            best_val_metric = primary
            torch.save(ckpt, str(ckpt_dir / "best.ckpt"))

    # 최종 metric 반환
    final_metrics = {**val_metrics}
    return final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = train(config)

    # METRICS stdout 계약 (마지막 줄, 고정 형식)
    safe_metrics = {}
    for k, v in metrics.items():
        try:
            safe_metrics[k] = float(v)
        except (TypeError, ValueError):
            safe_metrics[k] = 0.0
    print(f"METRICS:{json.dumps(safe_metrics)}")


if __name__ == "__main__":
    main()
