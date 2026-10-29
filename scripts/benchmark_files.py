"""Descriptive result names and non-overwriting benchmark run directories."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path

NAMES = {
    "table1": "square-breakdown",
    "table2": "field-primitives",
    "table3": "constant-point-addition",
    "table4": "scalar-schedules",
    "table5": "odd-encoding-savings",
    "table6": "resource-comparison",
    "signed": "signed-scalar-comparison",
}


def timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")


def create_directory(root, table, output=None):
    name = NAMES[table]
    if output is not None:
        directory = Path(output).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        return directory
    parent = Path(root)/"runs/benchmarks"/name
    parent.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    attempt = 1
    while True:
        directory = parent/(stamp if attempt == 1 else f"{stamp}-{attempt}")
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            attempt += 1


def paths(directory, table):
    return {extension: Path(directory)/(NAMES[table]+"."+extension)
            for extension in ("json", "tex", "md")}


def digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
