import pandas as pd
import pytest

from plot_benchmark import create_plots, load_measurements


def test_load_measurements_excludes_warmup_but_keeps_errors(tmp_path):
    path = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {
                "phase": "warmup",
                "idx": 0,
                "latency_ms": 0.5,
                "status": "ok",
                "is_iqr_outlier": "",
            },
            {
                "phase": "measure",
                "idx": 0,
                "latency_ms": 1.0,
                "status": "ok",
                "is_iqr_outlier": "False",
            },
            {
                "phase": "measure",
                "idx": 1,
                "latency_ms": 2.0,
                "status": "error",
                "is_iqr_outlier": "",
            },
        ]
    ).to_csv(path, index=False)

    measurements = load_measurements(path)

    assert len(measurements) == 2
    assert measurements["status"].tolist() == ["ok", "error"]
    assert measurements["is_iqr_outlier"].tolist() == [False, False]


def test_create_plots_writes_png_and_vector_pdf(tmp_path):
    measurements = pd.DataFrame(
        {
            "phase": ["measure"] * 5,
            "idx": [0, 1, 2, 3, 4],
            "latency_ms": [1.0, 1.2, 1.1, 1.3, 4.0],
            "status": ["ok"] * 5,
            "is_iqr_outlier": [False, False, False, False, True],
        }
    )

    paths = create_plots(
        measurements,
        {"guardrail_passed": True, "max_error_rate": 0.01},
        tmp_path,
        "run_1",
    )

    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    assert {path.suffix for path in paths} == {".png", ".pdf"}


def test_create_plots_rejects_overwrite(tmp_path):
    measurements = pd.DataFrame(
        {
            "phase": ["measure"] * 2,
            "idx": [0, 1],
            "latency_ms": [1.0, 2.0],
            "status": ["ok", "ok"],
            "is_iqr_outlier": [False, False],
        }
    )
    existing = tmp_path / "run_1_latency_ecdf.png"
    existing.touch()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_plots(measurements, {}, tmp_path, "run_1")
