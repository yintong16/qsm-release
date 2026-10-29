import copy
import json

import pytest

from qsm.audit import signed_schedule as audit


def test_terminal_measurements_use_the_active_curve(monkeypatch):
    import random
    from qsm.point import active
    assert audit.curve is active
    monkeypatch.setattr(audit, "add_const_affine_inplace", lambda circ, x, y, *args, **kw: (x, y))
    case = audit.terminal_case(255)
    expected = active.mul(1 << 255, active.GX, active.GY)
    assert case.meta["circuit_config"]["point"] == expected
    assert case.meta["circuit_config"]["curve"]["a"] == active.CURVE.A
    assert case.meta["circuit_config"]["curve"]["order"] == active.ORDER
    assert active.is_on_curve(*expected)
    values = case.gen_input(random.Random(41))
    assert active.is_on_curve(values["x"], values["y"])


def _archives():
    windows = {"version": "signed-window-v1", "field": "test", "source_digest": "fixed",
               "rows": [{"address_bits": 16, "strategy": strategy,
                         "result": {"observed_pass": True, "toffoli": cost}}
                        for strategy, cost in [("unsigned", 10), ("conjugate", 11)]]}
    terminals = {"version": "signed-terminals-v1", "field": "test", "source_digest": "fixed",
                 "rows": [{"exponent": 255, "result": {"observed_pass": True, "toffoli": 7}}]}
    return windows, terminals


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(audit, "source_digest", lambda: "fixed")
    monkeypatch.setattr(audit, "recoder_resource_report", lambda *args, **kwargs:
                        {"compute_uncompute": {"toffoli": 4}, "scope": "test recoder"})
    return _archives()


def test_ledger_charges_terminal_and_roundtrip_without_initialization_credit(model):
    result = audit.compose(*model, scalar_bits=[255, 256])
    a, b = [row["variants"] for row in result["rows"]]
    assert a["unsigned"]["ccx_model"] == 160
    assert a["carry"]["ccx_model"] == 15*11 + 7 + 4
    assert a["bias"]["ccx_model"] == 15*11 + 4
    assert b["carry"]["ccx_model"] == b["bias"]["ccx_model"] == 16*11 + 4
    assert "NOT an emitted full scalar circuit" in result["scope"]
    assert "no scalar-chain peak estimate" in result["scope"]
    assert "partial top" in result["scope"]


@pytest.mark.parametrize("mutation", ["field", "digest", "version", "failure", "missing_unsigned", "missing_terminal"])
def test_mismatched_or_missing_measurements_are_rejected(model, mutation):
    w, t = copy.deepcopy(model)
    if mutation == "field":
        t["field"] = "other"
    elif mutation == "digest":
        t["source_digest"] = "other"
    elif mutation == "version":
        w["version"] = "other"
    elif mutation == "failure":
        w["rows"][0]["result"]["observed_pass"] = False
    elif mutation == "missing_unsigned":
        w["rows"] = w["rows"][1:]
    else:
        t["rows"] = []
    with pytest.raises(ValueError):
        audit.compose(w, t, scalar_bits=[255])


def test_source_change_is_rejected(model, monkeypatch):
    monkeypatch.setattr(audit, "source_digest", lambda: "changed")
    with pytest.raises(ValueError, match="current source"):
        audit.compose(*model)


@pytest.mark.parametrize("kwargs", [{"scalar_bits": []}, {"scalar_bits": [True]},
                                    {"scalar_bits": [0]}, {"strategy": "unsigned"}])
def test_bad_model_requests_rejected(model, kwargs):
    with pytest.raises(ValueError):
        audit.compose(*model, **kwargs)


@pytest.mark.parametrize("exponents,trials", [([], 1), ([True], 1), ([-1], 1), ([1], 0)])
def test_bad_terminal_requests_fail_before_building(exponents, trials):
    with pytest.raises(ValueError):
        audit.measure_terminals(exponents, trials=trials)


def test_model_cli_refuses_to_overwrite(model, tmp_path):
    w, t = model
    wp, tp, out = (tmp_path / name for name in ("windows.json", "terminals.json", "model.json"))
    wp.write_text(json.dumps(w))
    tp.write_text(json.dumps(t))
    args = ["compose", "--windows", str(wp), "--terminals", str(tp), "--scalar-bits", "255",
            "--json", str(out)]
    assert audit.main(args) == 0
    assert json.loads(out.read_text())["rows"][0]["variants"]["carry"]["terminal_ccx"] == 7
    with pytest.raises(SystemExit):
        audit.main(args)
