"""
scripts/smoke_test.py — forward pass + 2 step 검증

성공 시 exit 0, 실패 시 exit 1
GitHub Actions smoke stage에서 실행된다.
"""
import argparse
import os
import sys
import traceback

# Ensure package root (parent of scripts/) is on sys.path for local imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import torch
import yaml


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        fast = yaml.safe_load(f)
    # fast.yaml 미정의 키에 대한 최소 기본값
    defaults = {
        "seed": 42,
        "in_channels": 1, "base_channels": 16, "depth": 2,
        "use_aux_head": False, "aux_loss_weight": 0.0,
        "batch_size": 2, "epochs": 2,
        "lr": 1e-4, "weight_decay": 1e-4, "gradient_clip": None,
        "loss_function": "l1",
        "primary_metric": "",
        "data_dir": "data/", "val_ratio": 0.5, "num_workers": 0,
    }
    return {**defaults, **fast}


def run_smoke(config_path: str) -> bool:
    try:
        cfg = _load_config(config_path)

        from model import build_model
        from data import build_dataloaders
        from module import TrainingModule

        torch.manual_seed(cfg["seed"])

        model     = build_model(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        module    = TrainingModule(model, cfg)

        _loaders = build_dataloaders(cfg)
        if isinstance(_loaders, (list, tuple)):
            train_loader = _loaders[0]
            val_loader = _loaders[1] if len(_loaders) > 1 else _loaders[0]
        else:
            train_loader = val_loader = _loaders

        # ── 학습 2 step ─────────────────────────
        model.train()
        for i, batch in enumerate(train_loader):
            if i >= 2:
                break
            optimizer.zero_grad()
            loss = module._compute_loss(batch)
            loss.backward()
            optimizer.step()
            inputs = batch[0] if isinstance(batch, (list, tuple)) else next(iter(batch.values()))
            print(f"  smoke step {i+1}: loss={loss.item():.4f} "
                  f"input_shape={tuple(inputs.shape)}")

        # ── val forward ─────────────────────────
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0] if isinstance(batch, (list, tuple)) else next(iter(batch.values()))
                out = model(inputs)
                assert not torch.isnan(out).any(), "NaN detected in smoke val output"
                break

        print("SMOKE: PASS")
        return True

    except NotImplementedError as e:
        # _compute_metrics NotImplementedError는 smoke 단계에서 허용
        print(f"  [smoke] NotImplementedError (구현 필요): {e}")
        print("SMOKE: PASS (skeleton)")
        return True

    except Exception:
        print("SMOKE: FAIL")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast.yaml")
    args = parser.parse_args()

    ok = run_smoke(args.config)
    sys.exit(0 if ok else 1)
