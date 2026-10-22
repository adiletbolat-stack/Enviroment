#!/usr/bin/env python3
"""
Quick look at a raw benchmark log + selection of the measured records.
Replaces the hard-coded C:\\Users\\... version: it works on any machine and inside Docker.

Run from the REPOSITORY ROOT:
    python src/parse_logs.py                              # default: data/raw/test_run.jsonl
    python src/parse_logs.py data/raw/another_run.jsonl   # any other file
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "src")]          # find load_logs.py whether this file is in src/ or in the root
from load_logs import describe, load_jsonl              # noqa: E402

DEFAULT_PATH = Path("data/raw/test_run.jsonl")          # relative to the repository root, NOT an absolute C:\ path
MEASURE_PHASES = {"measure", "benchmark"}               # benchmark.py writes "measure"; some logs use "benchmark"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"File not found: {path.resolve()}")
        print("Current folder:", Path.cwd())
        found = sorted(Path("data/raw").glob("*.jsonl")) + sorted(Path("logs").glob("*.jsonl"))
        print("JSONL files I can see in data/raw and logs:", [str(p) for p in found] or "none")
        print("Run the script from the repository root, or pass the file path as an argument.")
        sys.exit(1)

    res = load_jsonl(str(path))
    describe(res, str(path))

    df = res.runs
    selected = df[df["phase"].isin(MEASURE_PHASES)].copy()
    print(f"\nRecords before filtering: {len(df)}")
    print(f"Kept phases {sorted(MEASURE_PHASES)}: {len(selected)} | removed (e.g. warm-up): {len(df) - len(selected)}")
    if selected.empty:
        print(f"WARNING: nothing matches {sorted(MEASURE_PHASES)}. Phases in this file: {df['phase'].unique().tolist()}")
        sys.exit(1)
    print("\nFirst five measured records:")
    print(selected.head().to_string())


if __name__ == "__main__":
    main()