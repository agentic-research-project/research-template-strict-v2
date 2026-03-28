"""
train.py — PyTorch Fabric 학습 진입점 (템플릿)

규칙 (experiments/CLAUDE.md §3 §4 준수):
- Fabric은 이 파일에서만 초기화
- 마지막 stdout 줄은 반드시: METRICS:{...json...}
- 모든 하이퍼파라미터는 configs/default.yaml에서 로드
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure package root is on sys.path for local imports
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
    """학습 실행 후 최종 metric dict 반환."""
    fabric = Fabric(
        accelerator=config.get("accelerator", "auto"),
        precision=config.get("precision", "bf16-mixed"),
        devices=config.get("devices", 1),
        strategy=config.get("strategy", "auto"),
    )
    fabric.launch()

    # ── 재현성 ──────────────────────────
    fabric.seed_everything(config["seed"])

    # ── 모델 / 옵티마이저 생성 ──────────
    model = build_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # ── Fabric wrap (순서 고정) ──────────
    model, optimizer = fabric.setup(model, optimizer)
    _loaders = build_dataloaders(config)
    if isinstance(_loaders, (list, tuple)):
        train_loader = _loaders[0]
        val_loader = _loaders[1] if len(_loaders) > 1 else _loaders[0]
    else:
        train_loader = val_loader = _loaders
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # ── LR Scheduler ─────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config.get("lr_min", 1e-6),
    )

    module = TrainingModule(model, config)

    # ── 아티팩트 디렉토리 ────────────────
    ckpt_dir = Path("artifacts/checkpoints")
    log_dir  = Path("artifacts/logs")
    met_dir  = Path("artifacts/metrics")
    for d in [ckpt_dir, log_dir, met_dir]:
        d.mkdir(parents=True, exist_ok=True)

    best_val_metric = float("-inf")
    epoch_log_path  = met_dir / "per_epoch_metrics.jsonl"

    # ── 학습 루프 ────────────────────────
    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        # train
        model.train()
        train_metrics = module.train_epoch(fabric, train_loader, optimizer)

        # validate
        model.eval()
        with torch.no_grad():
            val_metrics = module.val_epoch(fabric, val_loader)

        scheduler.step()

        epoch_row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": round(time.time() - t0, 1),
        }
        with open(epoch_log_path, "a") as f:
            f.write(json.dumps(epoch_row) + "\n")

        pm = config.get("primary_metric", "")
        primary = val_metrics.get(pm, 0.0)
        fabric.print(
            f"Epoch {epoch:3d}/{config['epochs']} "
            f"| train_loss={train_metrics.get('loss', 0):.4f} "
            f"| val_{pm}={primary:.4f}"
        )

        # 체크포인트
        fabric.save(
            str(ckpt_dir / "last.ckpt"),
            {"model": model, "optimizer": optimizer,
             "epoch": epoch, "config": config},
        )
        if primary > best_val_metric:
            best_val_metric = primary
            fabric.save(
                str(ckpt_dir / "best.ckpt"),
                {"model": model, "optimizer": optimizer,
                 "epoch": epoch, "config": config,
                 "best_metric": best_val_metric},
            )

    # ── 최종 metrics ─────────────────────
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    final_metrics = {
        **val_metrics,
        "params_M": round(param_count, 2),
    }
    return final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = train(config)

    # ── METRICS stdout 계약 (마지막 줄, 고정 형식) ──
    print(f"METRICS:{json.dumps({k: float(v) for k, v in metrics.items()})}")


if __name__ == "__main__":
    main()
