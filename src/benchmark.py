#!/usr/bin/env python3
"""
benchmark.py — идеальный шаблон бенчмаркинга целевой функции.

Гарантии:
  1. Warm-up     : адаптивный (до стационарности) + жёсткий лимит.
                   Warm-up строки СОХРАНЯЮТСЯ в лог (флаг in_final_metrics=False),
                   а не удаляются — для аудита и дифференциации от рабочих замеров.
  2. Ошибки      : перехватываются, записываются с типом и сообщением,
                   НЕ замалчиваются и НЕ обрушивают эксперимент.
  3. Изоляция I/O: тайминг-цикл содержит ТОЛЬКО timer + call + try/except +
                   append в список. Никакого I/O, print, logging внутри.
  4. GC          : отключён на время тайминг-цикла; восстанавливается в finally.
  5. JSONL-лог   : meta-строка → N run-строк → summary-строка.
                   Временны́е метки: wall UTC с микросекундами, монотонный offset_ns.
  6. Метаданные  : CPU, RAM, ОС, Python, пакеты, git, cgroup — через env_metadata.
  7. Конфиг      : --config configs/config.yaml; CLI-флаги имеют приоритет.
  8. Перцентили  : p50, p95, p99 (и все остальные) по OK-замерам; ошибки — отдельно.

Использование:
    python src/benchmark.py --n-runs 1000 --seed 42 --out-dir data/raw
    python src/benchmark.py --config configs/config.yaml
    python src/benchmark.py --config configs/config.yaml --n-runs 200 --experiment smoke_test

Адаптация: отредактируй только секцию «EDIT THIS» (make_input + target).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# env_metadata — та же директория; при запуске из корня через python src/benchmark.py
# sys.path корректируется ниже.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import env_metadata as _em
    _HAS_ENV_METADATA = True
except ImportError:
    _HAS_ENV_METADATA = False

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ════════════════════════════════════════════════════════════════
# 1. EDIT THIS: ваша целевая функция
# ════════════════════════════════════════════════════════════════

def make_input(seed: int):
    """
    Setup: запускается ОДИН РАЗ, вне тайминга.
    Загружайте файлы, создавайте объекты, читайте БД — здесь.
    """
    rng = random.Random(seed)
    return [rng.random() for _ in range(20_000)]


def target(payload):
    """
    Целевая функция. Только чистые вычисления:
    • нет print/logging
    • нет записи в файлы
    • нет сетевых вызовов
    """
    return sorted(payload)[len(payload) // 2]

# ════════════════════════════════════════════════════════════════
# 2. Утилиты времени
# ════════════════════════════════════════════════════════════════

def utc_iso_us(wall_ns: int) -> str:
    """UTC ISO-8601 с МИКРОСЕКУНДАМИ через целочисленную арифметику (без float-погрешностей)."""
    base = datetime.fromtimestamp(wall_ns // 1_000_000_000, tz=timezone.utc)
    return (base + timedelta(microseconds=(wall_ns % 1_000_000_000) // 1_000)).isoformat(timespec="microseconds")


# ════════════════════════════════════════════════════════════════
# 3. Тайминг-цикл (I/O-изолирован)
# ════════════════════════════════════════════════════════════════

class TooManyErrors(RuntimeError):
    pass


def run_phase(
    func,
    payload,
    n: int,
    phase: str,
    first_idx: int,
    base_ns: int,
    rows: list,
    max_consecutive_errors: int = 20,
) -> None:
    """
    Тайминг-цикл. Единственные операции внутри:
        wall = time.time_ns()          ← wall-clock ДО старта perf_counter
        t0   = time.perf_counter_ns()  ← монотонный таймер START
        func(payload)                  ← ТОЛЬКО целевая функция
        t1   = time.perf_counter_ns()  ← монотонный таймер STOP
        rows.append(...)               ← запись в RAM, не в файл

    Больше НИЧЕГО. GC, I/O, logging — снаружи.
    """
    consecutive = 0
    for i in range(n):
        wall = time.time_ns()                   # wall-clock метка (для UTC-timestamp в JSONL)
        t0 = time.perf_counter_ns()
        err = None
        try:
            func(payload)
        except Exception as e:                  # noqa: BLE001 — ошибки = данные, не шум
            err = e
        t1 = time.perf_counter_ns()

        rows.append((phase, first_idx + i, wall, t0 - base_ns, t1 - t0, err))

        consecutive = consecutive + 1 if err else 0
        if consecutive >= max_consecutive_errors:
            raise TooManyErrors(
                f"{consecutive} подряд ошибок в фазе '{phase}' — аварийная остановка"
            )


# ════════════════════════════════════════════════════════════════
# 4. Warm-up: адаптивный до стационарности
# ════════════════════════════════════════════════════════════════

def _is_stationary(lat: list[float], window: int, cv_max: float, drift_max: float) -> bool:
    """
    Стационарность = CV последнего окна < cv_max
                   И дрейф между двумя последними окнами < drift_max.
    """
    if len(lat) < 2 * window:
        return False
    last = lat[-window:]
    prev = lat[-2 * window:-window]
    m_last, m_prev = statistics.mean(last), statistics.mean(prev)
    return (statistics.stdev(last) / m_last < cv_max
            and abs(m_last - m_prev) / m_prev < drift_max)


def adaptive_warmup(
    func,
    payload,
    base_ns: int,
    rows: list,
    min_iters: int = 20,
    max_iters: int = 500,
    step: int = 5,
    cv_max: float = 0.05,
    drift_max: float = 0.05,
) -> tuple[int, bool]:
    """
    Прогрев малыми шагами (step) до стационарности латентности.
    Возвращает (n_warmup_iters, converged).
    converged=False логируется явно — никогда не замалчивается.
    """
    done = 0
    while done < max_iters:
        run_phase(func, payload, step, "warmup", done, base_ns, rows)
        done += step

        # fail-fast если весь последний step — ошибки
        tail = rows[-step:]
        if all(r[5] is not None for r in tail):
            raise TooManyErrors(f"{step} подряд ошибок при warm-up")

        lat = [r[4] for r in rows if r[0] == "warmup" and r[5] is None]
        if done >= min_iters and _is_stationary(lat, window=10, cv_max=cv_max, drift_max=drift_max):
            return done, True

    return done, False      # не сошёлся — отчёт обязателен


# ════════════════════════════════════════════════════════════════
# 5. Статистика: перцентили, ошибки, SLO
# ════════════════════════════════════════════════════════════════

def _percentile(sorted_vals: list[float], p: float) -> float:
    """
    Линейная интерполяция (метод «fraction of index»).

    Почему не mean?
    ──────────────
    Среднее арифметическое катастрофически чувствительно к выбросам:
    один таймаут (например, 10 000 мс) сдвигает среднее для 1 000 замеров
    на 10 мс, маскируя реальное поведение системы.

    p95/p99 гарантируют: 95 % (99 %) запросов уложились в это время.
    Это прямая метрика пользовательского опыта (SLO).

    Медиана (p50) робастна к выбросам и позволяет сравнивать
    «типичное» поведение систем без влияния хвоста распределения.
    """
    k = (len(sorted_vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize(rows: list, slo_ms: float | None = None) -> dict:
    """
    Статистика только по рабочим прогонам (phase == 'measure').
    Ошибки НИКОГДА не удаляются — они получают собственную секцию.

    Аномалии vs шум:
    ─────────────────
    • Реальный спайк нагрузки: длительный (сотни мс), коррелирует с
      внешними событиями (load_avg_after > load_avg_before), воспроизводим.
    • GC-пауза: кратковременный (единицы мс), повторяется с определённой
      периодичностью, исчезает после gc.disable().
    • Аппаратный шум (TurboBoost, SMI): хаотичен, не коррелирует с нагрузкой.
    Для разграничения: сортируй по t_mono_ns и смотри автокорреляцию выбросов.
    """
    meas = [r for r in rows if r[0] == "measure"]
    ok_ms  = sorted(r[4] / 1e6 for r in meas if r[5] is None)
    err_ms = sorted(r[4] / 1e6 for r in meas if r[5] is not None)
    err_types: dict[str, int] = {}
    for r in meas:
        if r[5] is not None:
            k = type(r[5]).__name__
            err_types[k] = err_types.get(k, 0) + 1

    out: dict = {
        "type": "summary",
        "n_measured": len(meas),
        "n_ok": len(ok_ms),
        "error_rate": round(1 - len(ok_ms) / len(meas), 6) if meas else None,
        "errors_by_type": err_types,
    }

    if len(ok_ms) >= 2:
        out["ok_ms"] = {
            "mean":   round(statistics.mean(ok_ms), 4),
            "median": round(_percentile(ok_ms, 50), 4),   # p50 — робастная центральная тенденция
            "stdev":  round(statistics.stdev(ok_ms), 4),
            "p95":    round(_percentile(ok_ms, 95), 4),   # SLO-граница для 95 % запросов
            "p99":    round(_percentile(ok_ms, 99), 4),   # SLO-граница для 99 % запросов
            "min":    round(ok_ms[0], 4),
            "max":    round(ok_ms[-1], 4),
        }

    if err_ms:
        # fast-fail и timeout ведут себя ПРИНЦИПИАЛЬНО иначе → медиана ошибочных замеров
        out["error_time_to_failure_ms_median"] = round(_percentile(err_ms, 50), 4)

    if slo_ms is not None and meas:
        # ошибки ЗАСЧИТЫВАЮТСЯ как нарушение SLO (консервативная оценка)
        out["slo_ms"] = slo_ms
        out["slo_attainment"] = round(sum(1 for v in ok_ms if v <= slo_ms) / len(meas), 6)

    return out


# ════════════════════════════════════════════════════════════════
# 6. I/O: запись JSONL (ТОЛЬКО после измерений)
# ════════════════════════════════════════════════════════════════

def write_log(path: str, meta: dict, rows: list, summary: dict) -> None:
    """
    Весь I/O сосредоточен здесь. Вызывается ПОСЛЕ последнего замера.
    Режим 'x' (exclusive create): аварийный стоп при попытке перезаписи.

    Структура файла (одна строка = один JSON-объект):
        {"type": "meta",    ...}            ← метаданные эксперимента + окружения
        {"type": "run",     ...}  × N       ← один замер (warmup или measure)
        {"type": "summary", ...}            ← агрегированная статистика
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for phase, idx, wall, start_off, lat, err in rows:
            f.write(json.dumps({
                "type":            "run",
                "phase":           phase,
                "idx":             idx,
                "t_utc":           utc_iso_us(wall),  # wall-clock UTC с микросекундами
                "t_mono_ns":       start_off,          # монотонный offset от старта эксперимента
                "latency_ns":      lat,
                "status":          "ok" if err is None else "error",
                "error_type":      None if err is None else type(err).__name__,
                "error_msg":       None if err is None else str(err)[:200],
                "in_final_metrics": phase == "measure",  # warmup-строки помечены, не удалены
            }, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")


# ════════════════════════════════════════════════════════════════
# 7. Конфигурация: YAML + CLI override
# ════════════════════════════════════════════════════════════════

def _load_yaml_config(path: str) -> dict:
    if not _HAS_YAML:
        print("[WARN] PyYAML не установлен — --config игнорируется", file=sys.stderr)
        return {}
    with open(path, encoding="utf-8") as f:
        return _yaml.safe_load(f) or {}


def _merge_config(yaml_cfg: dict, args: argparse.Namespace) -> argparse.Namespace:
    """CLI-флаги имеют приоритет над config.yaml."""
    bm = yaml_cfg.get("benchmark", {})
    wu = yaml_cfg.get("warmup", {})
    slo = yaml_cfg.get("slo", {})

    defaults = {
        "n_runs":    bm.get("n_runs", 1000),
        "seed":      bm.get("seed", 42),
        "experiment": bm.get("experiment", "benchmark"),
        "out_dir":   bm.get("out_dir", "logs"),
        "slo_ms":    slo.get("latency_ms", None),
        "warmup_min": wu.get("min_iters", 20),
        "warmup_max": wu.get("max_iters", 500),
        "warmup_step": wu.get("step", 5),
        "warmup_cv":  wu.get("cv_max", 0.05),
        "warmup_drift": wu.get("drift_max", 0.05),
    }
    for k, v in defaults.items():
        if getattr(args, k, None) is None:  # CLI не задал — берём из YAML
            setattr(args, k, v)
    return args


# ════════════════════════════════════════════════════════════════
# 8. Точка входа
# ════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark harness с warm-up, JSONL-логом и перцентилями")
    ap.add_argument("--config",     default=None,     help="путь к config.yaml")
    ap.add_argument("--n-runs",     type=int,         default=None)
    ap.add_argument("--seed",       type=int,         default=None)
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--out-dir",    default=None)
    ap.add_argument("--slo-ms",     type=float,       default=None,
                    help="SLO-порог в мс; ошибки считаются нарушением")
    a = ap.parse_args()

    # Загружаем YAML, затем CLI перекрывает
    yaml_cfg = _load_yaml_config(a.config) if a.config else {}
    a = _merge_config(yaml_cfg, a)

    # Фиксация случайности
    random.seed(a.seed)
    try:
        import numpy as np
        np.random.seed(a.seed)
    except ImportError:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(a.seed))

    # Setup вне тайминга
    payload = make_input(a.seed)

    run_id = uuid.uuid4().hex[:8]
    rows: list = []
    status = "completed"

    # Метаданные окружения (CPU, RAM, OS, git, пакеты …)
    env = _em.collect(seed=a.seed, config=yaml_cfg) if _HAS_ENV_METADATA else {}

    meta: dict = {
        "type":        "meta",
        "run_id":      run_id,
        "experiment":  a.experiment,
        "seed":        a.seed,
        "n_runs":      a.n_runs,
        "started_utc": utc_iso_us(time.time_ns()),
        "argv":        sys.argv[1:],
        **env,           # разворачиваем всё из env_metadata
    }

    # ── Тайминг-цикл (GC выключен) ──────────────────────────────
    base_ns = time.perf_counter_ns()
    gc.collect()
    gc.disable()
    try:
        warm_n, converged = adaptive_warmup(
            target, payload, base_ns, rows,
            min_iters=a.warmup_min,
            max_iters=a.warmup_max,
            step=a.warmup_step,
            cv_max=a.warmup_cv,
            drift_max=a.warmup_drift,
        )
        meta.update(warmup_iterations=warm_n, warmup_converged=converged)
        if not converged:
            print(f"[WARN] Warm-up не сошёлся за {warm_n} итераций — данные сохранены, но результат под вопросом",
                  file=sys.stderr)
        run_phase(target, payload, a.n_runs, "measure", 0, base_ns, rows,
                  max_consecutive_errors=20)
    except (TooManyErrors, KeyboardInterrupt) as e:
        status = f"aborted:{type(e).__name__}"
    finally:
        gc.enable()

    meta["status"] = status

    # ── Статистика и I/O ─────────────────────────────────────────
    summary = summarize(rows, a.slo_ms)
    path = os.path.join(a.out_dir, f"{a.experiment}_{run_id}.jsonl")
    write_log(path, meta, rows, summary)

    print(f"log      : {path}")
    print(f"status   : {status}")
    print(f"warm-up  : {meta.get('warmup_iterations')} iter (converged={meta.get('warmup_converged')})")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
