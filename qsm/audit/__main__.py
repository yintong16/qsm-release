"""CLI: python -m qsm.audit [rates|refs|baselines|curves|preconditions|port|all]

`all` skips `port`, which forks a process per field and takes minutes; name it explicitly.
"""

from __future__ import annotations

import sys

from . import baselines, curves, port, preconditions, rates, refs

TOOLS = {"rates": rates.run, "refs": refs.run, "baselines": baselines.run,
         "curves": curves.run, "preconditions": preconditions.run,
         # Not in `all`: it forks four child processes and takes minutes. Run it by name.
         "port": port.run}

#: `all` runs everything cheap. `port` is opt-in for that reason.
SLOW = {"port"}


def main(argv: list[str]) -> int:
    which = argv[1:] or ["all"]
    if which == ["all"]:
        which = [t for t in TOOLS if t not in SLOW]
    bad = 0
    for name in which:
        if name not in TOOLS:
            print(f"unknown audit {name!r}; choose from {', '.join(TOOLS)} or 'all'")
            return 2
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        bad += TOOLS[name]() or 0
    print(f"\n{'=' * 78}")
    print("clean" if not bad else f"{bad} discrepancies -- see MISMATCH rows above")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
