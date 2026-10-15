"""
Automatic collection of experiment metadata (hardware, OS, software, container limits, code state).
Standard library only; psutil is used when installed for better cross-platform hardware data.
Every field is collected inside try/except: a missing value becomes None, never a crash.
"""
import hashlib
import json
import locale
import os
import platform
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata


def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:                                                    # noqa: BLE001
        return None


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:                                                    # noqa: BLE001
        return None


def cpu_info():
    info = {"machine": platform.machine(), "logical_cores": os.cpu_count(), "model": None,
            "physical_cores": None, "freq_mhz_max": None, "freq_mhz_current": None, "governor": None}
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        f = psutil.cpu_freq()
        if f:
            info["freq_mhz_max"], info["freq_mhz_current"] = f.max, f.current
    except ImportError:
        pass
    if platform.system() == "Linux":
        txt = _read("/proc/cpuinfo") or ""
        for line in txt.splitlines():
            if line.lower().startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
                break
        info["governor"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    elif platform.system() == "Windows":
        info["model"] = _run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"]) or platform.processor()
    elif platform.system() == "Darwin":
        info["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    info["model"] = info["model"] or platform.processor() or None
    return info


def memory_info():
    total = available = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        total, available = vm.total, vm.available
    except ImportError:
        for line in (_read("/proc/meminfo") or "").splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    return {"total_gb": round(total / 1e9, 2) if total else None,
            "available_gb": round(available / 1e9, 2) if available else None}


def container_info():
    """Docker/cgroup limits matter as much as the physical machine."""
    cg_cpu = _read("/sys/fs/cgroup/cpu.max")
    cg_mem = _read("/sys/fs/cgroup/memory.max")
    return {"in_container": os.path.exists("/.dockerenv") or bool(_read("/proc/1/cgroup") and "docker" in _read("/proc/1/cgroup")),
            "cgroup_cpu_max": cg_cpu, "cgroup_memory_max": cg_mem,
            "cpus_allowed": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def package_versions(names=("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib", "praw")):
    out = {}
    for n in names:
        try:
            out[n] = importlib_metadata.version(n)
        except importlib_metadata.PackageNotFoundError:
            out[n] = None
    return out


def file_sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:                                                    # noqa: BLE001
        return None


def git_state():
    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None,   # uncommitted changes = not reproducible
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])}


def collect(seed=None, config=None, lock_file="requirements.lock.txt"):
    return {
        "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version(),
               "platform": platform.platform(), "timezone": time.tzname[0], "locale": locale.getlocale()[0]},
        "cpu": cpu_info(),
        "ram": memory_info(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "compiler": platform.python_compiler(), "executable": sys.executable,
                   "hashseed": os.environ.get("PYTHONHASHSEED")},
        "packages": package_versions(),
        "lock_file_sha256": file_sha256(lock_file),
        "container": container_info(),
        "git": git_state(),
        "power": {"load_avg": os.getloadavg() if hasattr(os, "getloadavg") else None},
        "seed": seed,
        "config": config,
    }


if __name__ == "__main__":
    print(json.dumps(collect(seed=42, config={"n_runs": 1000}), indent=2))
