#!/usr/bin/env python3
"""
Real load test for Figure 1 (latency vs load): measures the RQ2 analysis step as a small local HTTP service
under several request rates (RPS). Nothing is simulated or invented: every row is a request that was really sent.

Design (so the numbers are honest):
  * OPEN-LOOP load: requests are sent on a fixed schedule (rate = RPS) whether or not earlier ones finished.
  * Latency is measured from the INTENDED send time, so queueing under overload is not hidden (no coordinated omission).
  * Server and load generator are separate processes; a warm-up phase is sent first and discarded.
  * Errors (timeouts, HTTP != 200) are kept in the table with status "error", never dropped.
  * One TCP connection per request (the simple stdlib server speaks HTTP/1.0), included in the latency.
  * Client and server run on the SAME machine and compete for CPU: read the curve as relative, not as absolute capacity.

Run from the repository root (pandas, numpy and scipy must be installed):
    python src/load_test.py
    python src/load_test.py --levels 10,25,50,100 --duration 20
Then:
    python src/make_figures.py --measurements data/processed/measurements_load.csv --only 1
"""
import argparse
import http.client
import json
import multiprocessing as mp
import socket
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ------------------------------------------------------------------ server (separate process)
def server_main(port, seed, ready):
    import bench_rq2                                  # the analysis step under test + synthetic coded table
    rows = bench_rq2.make_input(seed)                 # prepared once, outside any request

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                out = bench_rq2.target(rows)
                body, code = json.dumps({"status": out["status"]}).encode(), 200
            except Exception as e:                    # noqa: BLE001
                body, code = str(e).encode(), 500
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):                 # no console output inside the measured path
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    ready.set()
    srv.serve_forever()


# ------------------------------------------------------------------ load generator
def one_request(port, t_intended, wall_intended_ns, timeout, store, idx, rps):
    status, err = "ok", None
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        conn.request("GET", "/analyze")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        if resp.status != 200:
            status, err = "error", f"HTTP {resp.status}"
    except Exception as e:                            # noqa: BLE001 - timeouts and resets are data
        status, err = "error", type(e).__name__
    latency_s = time.perf_counter() - t_intended      # from the INTENDED start
    store.append((idx, rps, wall_intended_ns, latency_s * 1000.0, status, err))


def run_level(port, rps, duration, workers, timeout):
    n = int(rps * duration)
    store = []
    start = time.perf_counter() + 0.2
    wall0 = time.time_ns() + 200_000_000
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(n):
            t_i = start + i / rps
            while True:                                # sleep while far away, spin when close (timer granularity)
                dt = t_i - time.perf_counter()
                if dt <= 0:
                    break
                if dt > 0.02:
                    time.sleep(dt - 0.015)
            pool.submit(one_request, port, t_i, wall0 + int(i / rps * 1e9), timeout, store, i, rps)
    return store


def utc_iso_us(ns):
    base = datetime.fromtimestamp(ns // 1_000_000_000, tz=timezone.utc)
    return (base + timedelta(microseconds=(ns % 1_000_000_000) // 1000)).isoformat(timespec="microseconds")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="10,25,50,100,150", help="request rates in req/s, comma separated")
    ap.add_argument("--duration", type=float, default=15.0, help="seconds per level")
    ap.add_argument("--warmup-requests", type=int, default=100)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--experiment", default="load_test")
    ap.add_argument("--out", default="data/processed/measurements_load.csv")
    a = ap.parse_args()
    levels = [float(x) for x in a.levels.split(",")]

    port, ready = free_port(), mp.Event()
    server = mp.Process(target=server_main, args=(port, a.seed, ready), daemon=True)
    server.start()
    if not ready.wait(60):
        sys.exit("ERROR: the server did not start within 60 s")

    run_id = uuid.uuid4().hex[:8]
    print(f"run_id={run_id} | server on port {port} | levels (req/s): {levels} | {a.duration:g} s each")
    print(f"warm-up: {a.warmup_requests} requests at 20 req/s (discarded)")
    run_level(port, 20, a.warmup_requests / 20, a.workers, a.timeout)

    records = []
    for rps in levels:
        store = run_level(port, rps, a.duration, a.workers, a.timeout)
        lat = sorted(r[3] for r in store if r[4] == "ok")
        n_err = sum(r[4] == "error" for r in store)
        if lat:
            q = lambda p: lat[min(len(lat) - 1, int(p / 100 * len(lat)))]
            print(f"  {rps:6g} req/s: n={len(store):4d} errors={n_err:3d} | p50={q(50):8.2f} p95={q(95):8.2f} p99={q(99):8.2f} ms")
        else:
            print(f"  {rps:6g} req/s: n={len(store)} errors={n_err}: no successful request")
        records += [{"phase": "measure", "idx": len(records) + k, "t_utc": utc_iso_us(w), "rps": r, "latency_ms": l,
                     "status": s, "error_type": e, "run_id": run_id, "experiment": a.experiment}
                    for k, (i, r, w, l, s, e) in enumerate(store)]
        time.sleep(1.0)                                # let the server drain between levels
    server.terminate()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out, index=False)
    try:                                               # environment metadata next to the data (reproducibility)
        from env_metadata import collect
        meta = collect(seed=a.seed, config=vars(a))
        meta["run_id"] = run_id
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:                             # noqa: BLE001
        print("note: metadata not written:", e)
    print(f"\nsaved {len(records)} rows to {out}")


if __name__ == "__main__":
    mp.freeze_support()
    main()