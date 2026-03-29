"""
tests/test_forward.py — shape/dtype 단위 테스트 (GPT 제안 가능)
"""
import torch
import pytest
import yaml
from model import build_model


@pytest.fixture
def config():
    with open("configs/fast.yaml") as f:
        fast = yaml.safe_load(f)
    defaults = {
        "in_channels": 1, "base_channels": 16, "depth": 2,
        "use_aux_head": False, "seed": 42, "param_budget_M": 5.0,
    }
    return {**defaults, **fast}


def _make_input(config: dict) -> torch.Tensor:
    """config["input_shape"] 또는 기본값으로 더미 입력을 생성한다."""
    shape = config.get("input_shape", [config.get("in_channels", 1), 64, 64])
    return torch.randn(2, *shape)  # batch=2


def test_forward_no_crash(config):
    """forward pass가 예외 없이 완료되는지 확인."""
    model = build_model(config)
    x = _make_input(config)
    model.eval()
    with torch.no_grad():
        _ = model(x)


def test_output_dtype(config):
    model = build_model(config)
    x = _make_input(config)
    with torch.no_grad():
        out = model(x)
    assert out.dtype == x.dtype, f"Expected dtype {x.dtype}, got {out.dtype}"


def test_param_budget(config):
    """params_M ≤ config.get('param_budget_M', 5.0)"""
    model = build_model(config)
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    budget = config.get("param_budget_M", 5.0)
    assert params_M <= budget, f"params {params_M:.2f}M > budget {budget}M"


def test_no_nan_in_output(config):
    model = build_model(config)
    x = _make_input(config)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "NaN detected in model output"
