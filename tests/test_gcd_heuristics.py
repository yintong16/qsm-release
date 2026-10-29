import math
from pathlib import Path
import random
import runpy
import subprocess
import sys

import pytest

from qsm import params
from qsm.inversion.reference import dialog_widths_jump


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gcd_heuristics.py"
diagnostic = runpy.run_path(str(SCRIPT))
trace_widths = diagnostic["trace_widths"]
fitted_decrease = diagnostic["fitted_decrease"]
measure = diagnostic["measure"]


@pytest.mark.parametrize("widths,expected", [
    ([19, 16, 13, 10], 3),
    ([119, 116, 113, 110], 3),
    ([7, 7, 7], 0),
    ([1, 2, 3], -1),
    ([10, 8, 9, 4], 1.7),
])
def test_free_intercept_least_squares(widths, expected):
    assert fitted_decrease(widths) == pytest.approx(expected)


@pytest.mark.parametrize("widths", [[], [255]])
def test_slope_requires_two_observations(widths):
    with pytest.raises(ValueError, match="at least two"):
        fitted_decrease(widths)


@pytest.mark.parametrize("jump", [1, 2])
def test_trajectories_match_reference_exhaustively(jump):
    for p in (7, 31, 127):
        for x in range(1, p):
            widths = trace_widths(p, x, jump)
            assert widths == dialog_widths_jump(p, x, len(widths), jump)
            assert widths[0] == p.bit_length()
            assert dialog_widths_jump(p, x, len(widths) + 1, jump)[-1] == 1


def test_fullwidth_trajectories_match_reference():
    p = (1 << 255) - 19
    rng = random.Random(19)
    for x in [1, 2, p - 1, 1 << 254] + [rng.randrange(1, p) for _ in range(40)]:
        widths = trace_widths(p, x)
        assert widths == dialog_widths_jump(p, x, len(widths), 2)
        assert dialog_widths_jump(p, x, len(widths) + 1, 2)[-1] == 1


def test_fixed_seed_width_measurement():
    result = measure("width")
    assert set(result) == {"width_rate"}
    assert result["width_rate"] == pytest.approx(1.06177751, abs=5e-9)
    assert round(result["width_rate"], 3) == params.GCD_J2_WIDTH_RATE


def test_iteration_statistics_match_existing_budget_test():
    count_iters = runpy.run_path(str(ROOT / "tests" / "test_params_gcd_budget.py"))["_iters"]
    rng = random.Random(31337)
    counts = [count_iters(params.PRIME, rng.randrange(1, params.PRIME), 2) for _ in range(3000)]
    mean = sum(counts) / len(counts)
    sd = math.sqrt(sum((t - mean) ** 2 for t in counts) / len(counts))
    result = measure("iterations", samples=3000)
    assert result == pytest.approx({"mean": mean, "sd": sd,
                                    "mean_per_bit": mean / 255,
                                    "sd_per_sqrt_bit": sd / math.sqrt(255)})


def test_population_sd_denominator(monkeypatch):
    traces = iter([[255, 2], [255, 4, 2, 1]])
    monkeypatch.setitem(measure.__globals__, "trace_widths", lambda p, x: next(traces))
    result = measure("iterations", samples=2)
    assert result["mean"] == 3
    assert result["sd"] == 1


@pytest.mark.parametrize("heuristic", ["width", "iterations"])
def test_cli_is_standalone_and_prints_only_requested_estimates(heuristic, tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), heuristic, "--samples", "10"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    expected = measure(heuristic, samples=10)
    assert "samples=10" in result.stdout
    assert "seed" not in result.stdout
    assert "Method:" in result.stdout
    if heuristic == "width":
        assert f"Width decrease = {expected['width_rate']:.8f}" in result.stdout
        assert "Mean/n" not in result.stdout
    else:
        assert f"Mean/n = {expected['mean_per_bit']:.8f}" in result.stdout
        assert f"sd/sqrt(n) = {expected['sd_per_sqrt_bit']:.8f}" in result.stdout
        assert "Width decrease" not in result.stdout


def test_cli_defaults_are_reproducible(tmp_path):
    command = [sys.executable, "-I", str(SCRIPT), "width"]
    default = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    explicit = subprocess.run(command + ["--samples", "2000"],
                              cwd=tmp_path, capture_output=True, text=True, check=True)
    assert default.stdout == explicit.stdout


@pytest.mark.parametrize("samples", [0, 1, -1])
def test_cli_rejects_invalid_sample_counts(samples, tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "iterations", "--samples", str(samples)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "at least 2" in result.stderr
    with pytest.raises(ValueError, match="at least 2"):
        measure("iterations", samples=samples)


def test_unknown_heuristic_is_rejected():
    with pytest.raises(ValueError, match="heuristic must be"):
        measure("unknown")
