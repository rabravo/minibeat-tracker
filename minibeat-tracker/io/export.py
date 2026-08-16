"""CSV export — mirrors DataEvaluation export callbacks."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


def export_beating_data(path: Path, raw_results: np.ndarray, height_unit: str) -> None:
    """Export per-frame time series (mirrors _BeatingData.csv)."""
    cols = [
        "Time / sec",
        f"Contraction ({height_unit})",
        f"Y-Contraction ({height_unit})",
        f"X-Contraction ({height_unit})",
        "Contraction Angle (deg)",
    ]
    df = pd.DataFrame(raw_results[:, [0, 1, 3, 5, 7]], columns=cols)
    df.to_csv(path, index=False, float_format="%.4f")


def export_peaks(path: Path, peak_times: np.ndarray, peak_heights: np.ndarray, height_unit: str) -> None:
    """Export raw detected peaks (mirrors _RawPeaks.csv)."""
    df = pd.DataFrame({"Time / sec": peak_times, f"Height ({height_unit})": peak_heights})
    df.to_csv(path, index=False, float_format="%.4f")


def export_ana_peaks(
    path: Path,
    path_mean: Path,
    contract_heights: np.ndarray,
    relax_heights: np.ndarray,
    peaks_int_time_diff: np.ndarray,
    peaks_time_diff: np.ndarray,
    height_unit: str,
) -> None:
    """Export per-cycle analysis (mirrors _AnaPeaks.csv and _AnaPeaksMean.csv)."""
    beat_rates = 60.0 / peaks_time_diff
    # per-cycle
    n = min(len(contract_heights), len(relax_heights), len(peaks_int_time_diff))
    df = pd.DataFrame({
        f"Contraction velocity ({height_unit})": contract_heights[:n],
        f"Relaxation velocity ({height_unit})": relax_heights[:n],
        "Time difference / sec": peaks_int_time_diff[:n],
        "Beat rate / bpm": np.append(beat_rates, np.nan)[:n],
    })
    df.to_csv(path, index=False, float_format="%.4f")

    # summary
    summary = {
        "BeatRate_bpm_mean": [np.mean(beat_rates)],
        "BeatRate_bpm_std": [np.std(beat_rates)],
        "TimeInt_sec_mean": [np.mean(peaks_int_time_diff)],
        "TimeInt_sec_std": [np.std(peaks_int_time_diff)],
        f"MaxContract_{height_unit}_mean": [np.mean(contract_heights)],
        f"MaxContract_{height_unit}_std": [np.std(contract_heights)],
        f"MaxRelax_{height_unit}_mean": [np.mean(relax_heights)],
        f"MaxRelax_{height_unit}_std": [np.std(relax_heights)],
    }
    pd.DataFrame(summary).to_csv(path_mean, index=False, float_format="%.4f")
