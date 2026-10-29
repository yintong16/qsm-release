"""The opt-in optimized sweep counts physical gates, not compact IR records."""
import io
from types import SimpleNamespace

import pytest

from qsm.audit import window
from qsm.bench.registry import Case
from qsm.core.circuit import Circuit
from qsm.core.resources import count
from test_window_audit import tiny_case, fake_result


def test_compact_and_classical_accounting_matches_core():
    c = Circuit()
    control = c.input("j", 1)
    target = c.input("x", 3)
    start = len(c.ops)
    c.xor_word(tuple(target.ids), 7, ctrl=control[0])
    tmp = c.alloc(2)
    c.x(tmp[0])
    sources = tuple(c.measure(q) for q in tmp)
    base = c.classical_parities(sources, (0, 1, 2, 3))
    with c.condition(base + 3):
        c.z(control[0])
    case = Case(c, dict, lambda rng: {"j": 0, "x": 0}, {
        "resource_regions": [("unlookup", start, len(c.ops))],
    })
    measured = window.count_window_regions(case)
    r = count(c)
    assert measured["n_gates"] == r.n_gates == 7
    assert measured["cx"] == 3
    assert measured["clifford"] == r.clifford == 5
    assert measured["measurements"] == r.measurements == 2
    assert measured["peak_qubits"] == r.peak_qubits == 6
    assert measured["peak_qubits_by_region"]["unlookup"] == 6
    assert measured["classical_parity_rows"] == 4
    assert measured["classical_parity_sources"] == 2
    assert measured["classical_parity_bit_terms"] == 4
    assert "CPARITY" not in measured["gate_kinds"]
    assert "XOR_WORD" not in measured["gate_kinds"]


def test_sixteen_bits_supported_by_optimized_backend_without_override(monkeypatch):
    seen = []

    def build(bits, *, config):
        seen.append(config)
        return tiny_case(bits)

    monkeypatch.setattr(window, "build_optimized_window_case", build)
    monkeypatch.setattr(window, "benchmark_case", fake_result)
    result, = window.sweep([16], backend="optimized", operand_strategy="fused",
                           adder_chunk=64, fused_block=129, low_bits=7,
                           persist=False, stream=io.StringIO())
    cfg, = seen
    assert cfg.window_bits == 16 and cfg.operand_strategy == "fused"
    assert cfg.adder_chunk == 64 and cfg.fused_block == 129 and cfg.unlookup_low_bits == 7
    assert ".w16." in result.version


@pytest.mark.parametrize("options", [
    {"backend": "unknown"}, {"operand_strategy": "fused"},
    {"backend": "optimized", "operand_strategy": "unknown"},
    {"backend": "optimized", "adder_chunk": 0},
    {"backend": "optimized", "fused_block": None},
    {"backend": "optimized", "low_bits": -1},
    {"backend": "optimized", "low_bits": 5},
])
def test_bad_strategy_options_rejected_before_build(monkeypatch, options):
    monkeypatch.setattr(window, "build_window_case", lambda *a, **k: pytest.fail("built"))
    monkeypatch.setattr(window, "build_optimized_window_case", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(ValueError):
        window.sweep([4], **options)


def test_seventeen_bits_still_requires_override(monkeypatch):
    monkeypatch.setattr(window, "build_optimized_window_case", lambda *a, **k: pytest.fail("built"))
    with pytest.raises(ValueError, match="allow-large"):
        window.sweep([17], backend="optimized")


def test_cli_forwards_optimized_options(monkeypatch):
    def sweep(bits, **kw):
        assert bits == [16]
        assert kw["backend"] == "optimized" and kw["operand_strategy"] == "fused"
        assert kw["adder_chunk"] is None and kw["fused_block"] == 128
        assert kw["low_bits"] == 8 and kw["persist"] is False
        return [SimpleNamespace(observed_pass=True)]

    monkeypatch.setattr(window, "sweep", sweep)
    assert window.main(["--bits", "16", "--backend", "optimized", "--operand", "fused",
                        "--adder-chunk", "full", "--fused-block", "128", "--low-bits", "8",
                        "--no-persist"]) == 0


@pytest.mark.parametrize("choice", ["zero", "-1", "0"])
def test_invalid_cli_block(choice):
    with pytest.raises(SystemExit) as error:
        window.main(["--backend", "optimized", "--adder-chunk", choice])
    assert error.value.code == 2
