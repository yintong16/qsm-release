# QSM

Python implementations of reversible arithmetic and scalar-multiplication
circuits for elliptic-curve discrete logarithms, with Wei25519 as the primary
target and secp256k1 as a comparison.

The package constructs circuit operations, counts logical resources, and
provides value-, phase-, and ancilla-aware simulation tests. It does not require
a quantum processor, GPU, or external quantum SDK.

## Installation

Requirements: **Python 3.12 or newer**, NumPy and tabulate. Tests use pytest.
Dependencies are declared in [pyproject.toml](pyproject.toml). The setup helper
uses Python 3.12 by default and installs the runtime and test dependencies.

From this directory on macOS or Linux:

```bash
./setup.sh
```

For another installed interpreter:

```bash
PYTHON=python3.13 ./setup.sh
```

The helper creates or reuses `.venv`. The `paper` wrapper uses that environment
directly; activation is not required. A custom `VENV_DIR` is supported by the
setup helper, but the wrapper expects the default `.venv` location.

Manual installation:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m qsm info
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m qsm info
```

For the table commands on Windows, use
`.\.venv\Scripts\python.exe scripts/paper_tables.py table1`, replacing `table1`
with the required table. The POSIX shell wrappers are not Windows installers.

## Reproduce the manuscript tables

| Table | Command | Measurement |
|---|---|---|
| 1 | `./paper table1` | Square-subtraction component breakdown |
| 2 | `./paper table2` | Field-arithmetic primitives |
| 3 | `./paper table3` | Constant-point addition at default and matched GCD iteration budgets |
| 4 | `./paper table4` | Complete Wei25519 unsigned, zero-safe signed, and odd-only schedules |
| 5 | `./paper table5` | Odd-only encoding savings with matched arithmetic support |
| 6 | `./paper table6` | Fresh Wei25519 total with fixed published secp256k1 reference rows |

```bash
./paper --list
./paper table1 --out runs/review/square-breakdown
```

Every measured row is computed from newly constructed circuit operations.
No stored benchmark archive is required or used as a fallback. Independent
field or construction builds run in parallel. For Tables 1–3, progress messages
are shown during construction, but the complete result table is printed only
once all field workers have finished, validation has passed, and the output
files have been saved. Worker timing does not interleave messages with table
rows. Tables 4 and 6 emit complete scalar schedules rather than estimating their
totals by multiplying a point cost by the number of additions.

Full scalar builds can take tens of minutes and require several GB of memory
per worker. Table 4 starts three workers. Tables 3 and 5 build full-width point
circuits and are also substantially heavier than primitive measurements.

To rebuild only the zero-safe signed scalar comparison:

```bash
./paper signed                         # both fields concurrently
./paper signed --field 25519            # Wei25519 only
./paper signed --field secp256k1        # secp256k1 only
```

### Outputs

The table commands generate LaTeX, Markdown and JSON with these basenames:

| Table | Output basename |
|---|---|
| 1 | `square-breakdown` |
| 2 | `field-primitives` |
| 3 | `constant-point-addition` |
| 4 | `scalar-schedules` |
| 5 | `odd-encoding-savings` |
| 6 | `resource-comparison` |

Results are written under
`runs/benchmarks/<benchmark-name>/<YYYY-MM-DD_HH-MM-SSZ>/`, using UTC.
`--out` selects a new directory. Existing destinations are refused, and runs
started within the same second receive distinct directory suffixes.

JSON records retain the selected configurations, source and runner identities,
resource counts, timestamps, and applicable stage, component and diagnostic
information. Progress logs and per-construction records are also retained.
Failed builds and consistency checks do not fall back to earlier values.
Failed strict diagnostics prevent completed full-schedule tables from being
emitted; their raw results and logs remain available for inspection.

The signed-only command writes `signed-scalar-comparison.json` and a Markdown
summary, together with per-field records and logs.

Generated LaTeX uses `booktabs`; Tables 2–6 also use `tabularx`. Table 6 retains
`\cite{Schrottenloher26}` for the manuscript bibliography.

### Configurations and reference values

- The selected square circuits use a 40-bit narrow-row ripple margin in both
  fields. Modular-reduction windows remain 45 bits for Wei25519 and 73 for
  secp256k1.
- The full Wei25519 schedules use 253/208 scalar-control bits, radix 2^16,
  recycled controls, one initializer and 28 additions. The secp256k1 signed
  comparison uses 256/208 bits.
- The signed and odd-only rows in Table 4 use canonical-zero replay and
  canonical positive-coordinate restoration. The unsigned row uses its baseline
  arithmetic. Table 5 compares signed and odd-only additions with matched
  arithmetic support and verifies their exclusive component differences.
- Table 6 uses fixed literature quotations for secp256k1: space-optimized
  `2^26.11` Toffoli / 1,208 logical qubits, and gate-optimized `2^25.78` / 1,462.
  These rows are not remeasured by the command. Only the Wei25519 row is rebuilt.

## Reproduce the GCD heuristics

Run from this directory; Python's standard library is sufficient:

```bash
python3 scripts/gcd_heuristics.py width --samples 2000
python3 scripts/gcd_heuristics.py iterations --samples 2000
```

Both commands sample uniform nonzero elements modulo `p = 2^255 - 19` with
replacement and run jump-2 GCD. The default is 2,000 samples.
Each prints the requested estimates and their calculation method:

- **Width:** record `bit_length(max(u,v))` before each iteration, retain widths
  greater than one, and fit `width(i) = intercept - rate*i` with both parameters
  free for each input. Average the rates: **1.06177751**, rounding to **1.062**.
- **Iterations:** count iterations until `v = 0`; compute the mean and population
  standard deviation, then print `mean/n` and `sd/sqrt(n)`, with `n = 255`.

These are reproducible measurements, not a replay of the unarchived historical
samples. The width slope does not establish the padding's coverage.

## Command-line tools

The lower-level interface supports explicit field selection:

```bash
.venv/bin/python -m qsm --help
.venv/bin/python -m qsm info
.venv/bin/python -m qsm list
.venv/bin/python -m qsm --field secp256k1 info
.venv/bin/python -m qsm audit --help
.venv/bin/python -m qsm audit full-scalar --help
```

Use `python -m qsm` with the intended environment's interpreter when several
installations are present. Put `--field` before the command. Named audit tools
can construct large circuits; inspect their help before running them.

The optional signed point benchmark includes recoding and selected all-outcome
diagnostics without rebuilding a full scalar schedule:

```bash
.venv/bin/python -I -B scripts/bench_zero_safe_signed.py --field 25519 --json runs/signed-point.json
```

## Tests

The included tests are self-contained and do not require manuscript drafts,
experiment archives or reference repositories.

```bash
.venv/bin/python -m pytest -m 'not slow'
```

The `slow` marker identifies explicitly marked large-circuit tests. Some other
resource regressions also construct full-width arithmetic, so the broad
non-slow selection can take tens of minutes. Registry-wide construction sweeps
are marked `slow`. Running without a marker selection includes those additional
expensive tests.

For a focused interface and zero-safe integration check:

```bash
.venv/bin/python -m pytest tests/test_release_layout.py tests/test_signed_zero_safe.py tests/test_paper_table4.py tests/test_paper_table5.py tests/test_paper_table6.py
```

## Interpretation of results

Resource counts are logical CCX/Toffoli counts and peak simultaneously live
logical qubits, including temporary workspace. They are not physical
fault-tolerant hardware estimates. Feedback-rotation synthesis is unpriced,
and full-size depth is not computed by the default scalar benchmarks.

The tuned arithmetic is approximate. Zero-safe replay supports literal
canonical zero numerators/products with nonzero factors; it does not establish
universal exactness or support arbitrary noncanonical encodings. Tests retain
known value, phase and cleanup hazards. Selected basis and symbolic diagnostics
are not statistical success-probability bounds or full Shor statevector runs.
The six table commands do not perform a large statistical calibration.

## Package layout

```text
qsm/            circuit construction, arithmetic, simulation and resource tools
scripts/        table-generation and signed point-benchmark drivers
tests/          portable unit, integration and interface tests
paper           short table-command wrapper
setup.sh        environment setup
pyproject.toml  package metadata and dependencies
```

## License

No project license has been specified. Source-code citations and existing
attribution are retained; third-party reference implementations are not bundled.
