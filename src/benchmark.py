#!/usr/bin/env python3
"""
Benchmark template for a target function.

  1. warm-up      : adaptive (runs until the last timings are stationary), capped, logged, never in final metrics
  2. measurement  : n_runs timed calls; errors are CAPTURED and KEPT (no silent drops, no crash)
  3. I/O isolation: the timed loop touches only the timer and an in-memory list; ALL writing happens afterwards
  4. JSONL log    : meta line, one line per call (UTC timestamp with microseconds), summary line

Usage:  python src/benchmark.py --n-runs 1000 --seed 42 --out-dir logs
Edit only section 1 (make_input, target).
"""
import argparse
import gc
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone


# ============================ 1. EDIT THIS: input and target ============================
def make_input(seed: int):
    """Setup. Runs ONCE, outside every timed region (load data, build objects, read files here)."""
    rng = random.Random(seed)
    return [rng.random() for _ in range(20_000)]


def target(payload):
    """The function under test. Pure computation: no print, no logging, no file writes."""
    return sorted(payload)[len(payload) // 2]
# ========================================================================================


class TooManyErrors(RuntimeError):
    pass


def utc_iso_us(wall_ns: int) -> str:
    """UTC ISO-8601 with MICROSECONDS, built with integer arithmetic (no float rounding)."""
    base = datetime.fromtimestamp(wall_ns // 1_000_000_000, tz=timezone.utc)
    return (base + timedelta(microseconds=(wall_ns % 1_000_000_000) // 1000)).isoformat(timespec="microseconds")


def run_phase(func, payload, n, phase, first_idx, base_ns, rows, max_consecutive_errors=20):
    """Timed loop. Inside: timer, call, try/except, in-memory append. Nothing else."""
    consecutive = 0
    for i in range(n):
        wall = time.time_ns()                       # wall clock, taken BEFORE the timer starts
        t0 = time.perf_counter_ns()
        err = None
        try:
            func(payload)
        except Exception as e:                      # noqa: BLE001 - failures are data, not noise
            err = e
        t1 = time.perf_counter_ns()
        rows.append((phase, first_idx + i, wall, t0 - base_ns, t1 - t0, err))
        consecutive = consecutive + 1 if err else 0
        if consecutive >= max_consecutive_errors:   # fail fast, but the data collected so far is kept
            raise TooManyErrors(f"{consecutive} consecutive errors in phase '{phase}'")


def stationary(lat, window=10, cv_max=0.05, drift_max=0.05):
    if len(lat) < 2 * window:
        return False
    last, prev = lat[-window:], lat[-2 * window:-window]
    m_last, m_prev = statistics.mean(last), statistics.mean(prev)
    return statistics.stdev(last) / m_last < cv_max and abs(m_last - m_prev) / m_prev < drift_max


def adaptive_warmup(func, payload, base_ns, rows, min_iters=20, max_iters=500, step=5):
    """Warm up in small steps until stationary. Returns (iterations, converged)."""
    done = 0
    while done < max_iters:
        run_phase(func, payload, step, "warmup", done, base_ns, rows)
        done += step
        tail = rows[-20:]
        if len(tail) == 20 and all(r[5] is not None for r in tail):
            raise TooManyErrors("20 consecutive errors during warm-up")
        lat = [r[4] for r in rows if r[0] == "warmup" and r[5] is None]
        if done >= min_iters and stationary(lat):
            return done, True
    return done, False                              # not converged: must be reported, never hidden


def pct(sorted_vals, p):
    k = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize(rows, slo_ms=None):
    """Latency statistics use OK runs only; errors are never hidden: they get their own rate, types and timing."""
    meas = [r for r in rows if r[0] == "measure"]
    ok = sorted(r[4] / 1e6 for r in meas if r[5] is None)
    err_lat = sorted(r[4] / 1e6 for r in meas if r[5] is not None)
    errs = {}
    for r in meas:
        if r[5] is not None:
            errs[type(r[5]).__name__] = errs.get(type(r[5]).__name__, 0) + 1
    out = {"type": "summary", "n_measured": len(meas), "n_ok": len(ok),
           "error_rate": 1 - len(ok) / len(meas) if meas else None, "errors_by_type": errs}
    if len(ok) > 1:
        out["ok_ms"] = {"mean": statistics.mean(ok), "median": pct(ok, 50), "stdev": statistics.stdev(ok),
                        "p95": pct(ok, 95), "p99": pct(ok, 99), "min": ok[0], "max": ok[-1]}
    if err_lat:
        out["error_time_to_failure_ms_median"] = pct(err_lat, 50)   # fast-fail and timeout errors differ a lot
    if slo_ms is not None and meas:                                 # errors count as SLO violations
        out["slo_ms"] = slo_ms
        out["slo_attainment"] = sum(1 for v in ok if v <= slo_ms) / len(meas)
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:                               # noqa: BLE001
        return None


def write_log(path, meta, rows, summary):
    """ALL I/O happens here, after the last measurement. One new file per run, never overwritten."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:    # mode "x": fail instead of overwriting
        f.write(json.dumps(meta) + "\n")
        for phase, idx, wall, start_off, lat, err in rows:
            f.write(json.dumps({
                "type": "run", "phase": phase, "idx": idx,
                "t_utc": utc_iso_us(wall),                  # microsecond wall-clock timestamp
                "t_mono_ns": start_off,                     # monotonic offset from experiment start
                "latency_ns": lat,
                "status": "ok" if err is None else "error",
                "error_type": None if err is None else type(err).__name__,
                "error_msg": None if err is None else str(err)[:200],
                "in_final_metrics": phase == "measure",     # warm-up rows are flagged, not deleted
            }) + "\n")
        f.write(json.dumps(summary) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--experiment", default="benchmark")
    ap.add_argument("--out-dir", default="logs")
    ap.add_argument("--slo-ms", type=float, default=None, help="latency threshold; errors count as violations")
    a = ap.parse_args()

    random.seed(a.seed)
    try:
        import numpy as np
        np.random.seed(a.seed)
    except ImportError:
        pass

    payload = make_input(a.seed)                    # setup, outside timing
    run_id = uuid.uuid4().hex[:8]
    rows, status = [], "completed"
    meta = {"type": "meta", "run_id": run_id, "experiment": a.experiment, "seed": a.seed, "n_runs": a.n_runs,
            "started_utc": utc_iso_us(time.time_ns()), "git_commit": git_commit(),
            "python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "argv": sys.argv[1:],
            "load_avg_before": os.getloadavg() if hasattr(os, "getloadavg") else None}

    base_ns = time.perf_counter_ns()
    gc.collect()
    gc.disable()                                    # no GC pauses inside the timed loops
    try:
        warm_n, converged = adaptive_warmup(target, payload, base_ns, rows)
        meta.update(warmup_iterations=warm_n, warmup_converged=converged)
        run_phase(target, payload, a.n_runs, "measure", 0, base_ns, rows)
    except (TooManyErrors, KeyboardInterrupt) as e:
        status = f"aborted: {type(e).__name__}"      # keep whatever was collected
    finally:
        gc.enable()

    meta["status"] = status
    meta["load_avg_after"] = os.getloadavg() if hasattr(os, "getloadavg") else None
    summary = summarize(rows, a.slo_ms)
    path = os.path.join(a.out_dir, f"{a.experiment}_{run_id}.jsonl")
    write_log(path, meta, rows, summary)

    print(f"log: {path} | status: {status} | warm-up: {meta.get('warmup_iterations')} "
          f"(converged={meta.get('warmup_converged')})")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
