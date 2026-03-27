"""A-8: Summary layer diagnostic tests.

Tests for the diagnostic helpers in research_loop.py:
1. metrics_parse_error still has recommended_next_actions
2. mechanism_ok=false -> first action is Path A
3. plateau -> bottleneck_candidates generated
4. stable but under-target -> confidence != 0.9
5. repeated contradiction -> Path C prior
"""

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

from lab.research_loop import (
    _infer_training_stability,
    _infer_bottlenecks,
    _infer_recommended_actions,
    _estimate_confidence,
)


# ── Fixtures ────────────────────────────────────────────────

def _make_run_result(
    status: str = "success",
    metrics: dict | None = None,
    stdout_lines: list[str] | None = None,
) -> dict:
    return {
        "status": status,
        "metrics": metrics or {},
        "stdout_lines": stdout_lines or [],
        "stderr_tail": [],
        "returncode": 0 if status == "success" else 1,
        "metadata": {},
    }


def _make_primary(value: float, target: float, name: str = "psnr") -> dict:
    return {
        "name": name,
        "value": value,
        "target": target,
        "met": value >= target,
        "unit": "",
    }


def _make_history_entry(
    met: bool = False,
    loss_converged: bool = True,
    plateau_detected: bool = False,
) -> dict:
    return {
        "run_id": "test_v1",
        "version": 1,
        "status": "success",
        "metrics": {"name": "psnr", "value": 25.0, "target": 30.0, "met": met},
        "training_stability": {
            "loss_converged": loss_converged,
            "nan_detected": False,
            "plateau_detected": plateau_detected,
        },
    }


# ── Test 1: metrics_parse_error -> recommended_next_actions not empty ──

def test_metrics_parse_error_has_actions():
    """metrics_parse_error should still produce recommended_next_actions."""
    run_result = _make_run_result(status="metrics_parse_error")
    stability = _infer_training_stability(run_result, {})
    primary = _make_primary(0.0, 30.0)
    hypothesis_impl = {}

    bottlenecks = _infer_bottlenecks(stability, primary, hypothesis_impl, run_result)
    actions = _infer_recommended_actions(
        stability, primary, hypothesis_impl, bottlenecks, [], run_result,
    )

    assert len(actions) > 0, "metrics_parse_error should produce at least one action"
    assert actions[0]["path"] == "A", "first action should be Path A for execution failure"


# ── Test 2: mechanism_ok=false -> first action is Path A ──

def test_mechanism_not_implemented_path_a():
    """mechanism_audit.implemented=false should make first action Path A."""
    run_result = _make_run_result(status="success", metrics={"psnr": 20.0})
    stability = _infer_training_stability(run_result, {"psnr": 20.0})
    primary = _make_primary(20.0, 30.0)
    hypothesis_impl = {"mechanism_audit": {"implemented": False}}

    bottlenecks = _infer_bottlenecks(stability, primary, hypothesis_impl, run_result)
    actions = _infer_recommended_actions(
        stability, primary, hypothesis_impl, bottlenecks, [], run_result,
    )

    # Find the mechanism-related action
    mechanism_actions = [a for a in actions if "mechanism" in a.get("rationale", "").lower()]
    assert len(mechanism_actions) > 0, "should have mechanism-related action"
    assert mechanism_actions[0]["path"] == "A"
    assert mechanism_actions[0]["priority"] == "high"


# ── Test 3: plateau -> bottleneck_candidates generated ──

def test_plateau_generates_bottleneck():
    """Plateau detection should produce bottleneck candidates."""
    stdout_lines = [
        "epoch 1: loss=0.5, val_psnr=25.0",
        "epoch 2: loss=0.48, val_psnr=25.01",
        "epoch 3: loss=0.47, val_psnr=25.01",
        "epoch 4: loss=0.47, val_psnr=25.02",
    ]
    run_result = _make_run_result(
        status="success",
        metrics={"psnr": 25.02},
        stdout_lines=stdout_lines,
    )
    stability = _infer_training_stability(run_result, {"psnr": 25.02})
    primary = _make_primary(25.02, 30.0)

    # Force plateau_detected for test (in case regex doesn't match exact format)
    stability["plateau_detected"] = True

    bottlenecks = _infer_bottlenecks(stability, primary, {}, run_result)

    capacity_bottlenecks = [b for b in bottlenecks if "capacity" in b["name"].lower()]
    assert len(capacity_bottlenecks) > 0, "plateau should generate capacity bottleneck"


# ── Test 4: stable but under-target -> confidence != 0.9 ──

def test_stable_under_target_confidence_not_high():
    """Stable training but under-target should not give confidence 0.9."""
    run_result = _make_run_result(status="success", metrics={"psnr": 25.0})
    stability = {
        "loss_converged": True,
        "nan_detected": False,
        "lr_schedule_ok": True,
        "plateau_detected": False,
        "overfitting_suspected": False,
        "undertraining_suspected": False,
        "notes": [],
    }
    primary = _make_primary(25.0, 30.0)

    conf = _estimate_confidence(primary, stability, {}, {}, [], run_result)

    assert conf["final_confidence"] < 0.9, (
        f"confidence should be < 0.9 for under-target result, got {conf['final_confidence']}"
    )
    assert conf["final_confidence"] > 0.0, "confidence should not be zero for stable run"
    assert conf["explanation"], "explanation should not be empty"


# ── Test 5: repeated contradiction -> Path C prior ──

def test_repeated_contradiction_path_c():
    """3+ runs with mechanism implemented but target unmet should suggest Path C."""
    run_result = _make_run_result(status="success", metrics={"psnr": 22.0})
    stability = {
        "loss_converged": True,
        "nan_detected": False,
        "lr_schedule_ok": True,
        "plateau_detected": True,
        "overfitting_suspected": False,
        "undertraining_suspected": False,
        "notes": [],
    }
    primary = _make_primary(22.0, 30.0)
    hypothesis_impl = {"mechanism_audit": {"implemented": True}}

    # Build 3+ run history with all converged but unmet
    run_history = [
        _make_history_entry(met=False, loss_converged=True) for _ in range(3)
    ]

    bottlenecks = _infer_bottlenecks(stability, primary, hypothesis_impl, run_result)
    actions = _infer_recommended_actions(
        stability, primary, hypothesis_impl, bottlenecks, run_history, run_result,
    )

    path_c_actions = [a for a in actions if a["path"] == "C"]
    assert len(path_c_actions) > 0, (
        f"3+ contradicted runs with mechanism implemented should suggest Path C. "
        f"Got actions: {actions}"
    )


# ── Additional: training stability nan detection ──

def test_nan_detection():
    """NaN in stdout should be detected."""
    run_result = _make_run_result(
        status="success",
        metrics={"psnr": 0.0},
        stdout_lines=["epoch 1: loss=nan"],
    )
    stability = _infer_training_stability(run_result, {"psnr": 0.0})
    assert stability["nan_detected"] is True
    assert stability["loss_converged"] is False


def test_lr_schedule_absent():
    """No LR data should give lr_schedule_ok=None."""
    run_result = _make_run_result(
        status="success",
        metrics={"psnr": 25.0},
        stdout_lines=["epoch 1: loss=0.5"],
    )
    stability = _infer_training_stability(run_result, {"psnr": 25.0})
    assert stability["lr_schedule_ok"] is None, "no LR data should give None"


if __name__ == "__main__":
    if pytest:
        pytest.main([__file__, "-v"])
    else:
        # Run tests directly without pytest
        tests = [
            ("metrics_parse_error_has_actions", test_metrics_parse_error_has_actions),
            ("mechanism_not_implemented_path_a", test_mechanism_not_implemented_path_a),
            ("plateau_generates_bottleneck", test_plateau_generates_bottleneck),
            ("stable_under_target_confidence_not_high", test_stable_under_target_confidence_not_high),
            ("repeated_contradiction_path_c", test_repeated_contradiction_path_c),
            ("nan_detection", test_nan_detection),
            ("lr_schedule_absent", test_lr_schedule_absent),
        ]
        passed = 0
        for name, fn in tests:
            try:
                fn()
                print(f"  PASSED: {name}")
                passed += 1
            except AssertionError as e:  # noqa: F841
                print(f"  FAILED: {name}: {e}")
            except Exception as e:
                print(f"  ERROR:  {name}: {e}")
        print(f"\n{passed}/{len(tests)} tests passed")
