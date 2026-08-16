"""Contraction analysis — port of GetContractionData.m, calcMeanContraction.m,
doAutoCorr.m, findpeaks logic from DataEvaluation.m."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, correlate


# ---------------------------------------------------------------------------
# Step 3: Get Contraction Data
# ---------------------------------------------------------------------------

def get_contraction_data(
    motion_vect: np.ndarray,
    motion_amp: np.ndarray,
    delay: int = 2,
    framerate: float = 14.0,
    pixel_size: float = 0.0,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Collapse 4D motion field to N×9 per-frame time series.

    Columns
    -------
    0  time (sec)
    1  mean absolute amplitude
    2  std absolute amplitude
    3  mean |Y| motion
    4  std Y motion
    5  mean |X| motion
    6  std X motion
    7  mean contraction angle (deg)
    8  std contraction angle (deg)
    """
    R, C, _, T = motion_vect.shape

    # Active-block mask: blocks that moved at any time point
    amp_sum = motion_amp.sum(axis=2)       # [R, C]
    active = (amp_sum > 0).astype(np.float64)

    if mask is None:
        scaled_mask = np.ones((R, C), dtype=bool)
    else:
        # mask is at image resolution; scale down to block grid
        scale = mask.shape[0] / R
        scaled_mask = mask[::int(scale), ::int(scale)][:R, :C].astype(bool)

    combined = active * scaled_mask

    results = np.zeros((T, 9), dtype=np.float64)

    for t in range(T):
        results[t, 0] = t * delay / framerate

        amp_t = motion_amp[:, :, t] * combined
        results[t, 1] = np.nanmean(amp_t[combined > 0]) if combined.any() else 0.0
        results[t, 2] = np.nanstd(amp_t[combined > 0]) if combined.any() else 0.0

        vy = motion_vect[:, :, 0, t] * combined
        results[t, 3] = np.nanmean(np.abs(vy[combined > 0])) if combined.any() else 0.0
        results[t, 4] = np.nanstd(vy[combined > 0]) if combined.any() else 0.0

        vx = motion_vect[:, :, 1, t] * combined
        results[t, 5] = np.nanmean(np.abs(vx[combined > 0])) if combined.any() else 0.0
        results[t, 6] = np.nanstd(vx[combined > 0]) if combined.any() else 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            angle = np.degrees(np.arctan(np.abs(vy) / (np.abs(vx) + 1e-9)))
        results[t, 7] = np.nanmean(angle[combined > 0]) if combined.any() else 0.0
        results[t, 8] = np.nanstd(angle[combined > 0]) if combined.any() else 0.0

    # Zero angle during non-beating frames (amplitude below mean)
    mean_amp = results[:, 1].mean()
    results[results[:, 1] < mean_amp, 7:9] = np.nan

    # Unit conversion
    unit = "µm/sec" if pixel_size > 0 else "px/sec"
    if pixel_size > 0:
        results[:, 1:7] *= pixel_size * framerate / delay
    else:
        results[:, 1:7] *= framerate / delay

    legend = [
        "Time (sec)",
        f"Mean Abs Amplitude ({unit})", f"Std Abs Amplitude ({unit})",
        f"Mean |Y| Motion ({unit})", f"Std Y Motion ({unit})",
        f"Mean |X| Motion ({unit})", f"Std X Motion ({unit})",
        "Mean Angle (deg)", "Std Angle (deg)",
    ]
    return results, legend


# ---------------------------------------------------------------------------
# Spatial contraction maps (Step 4)
# ---------------------------------------------------------------------------

def calc_mean_contraction(
    motion_amp: np.ndarray,
    motion_vect: np.ndarray,
    mask: np.ndarray | None = None,
    cust_max: float = 0.0,
    thresh_frac: float = 0.2,
) -> dict:
    """Time-averaged spatial maps — port of calcMeanContraction.m."""
    mean_abs = motion_amp.mean(axis=2)          # [R, C]
    mean_y = motion_vect[:, :, 0, :].mean(axis=2)
    mean_x = motion_vect[:, :, 1, :].mean(axis=2)

    peak = cust_max if cust_max > 0 else mean_abs.max()
    thresh = thresh_frac * peak

    active_area = (mean_abs > thresh).sum()
    total_area = mean_abs.size

    frac_x = np.abs(mean_x).sum() / (np.abs(mean_abs).sum() + 1e-9)
    frac_y = np.abs(mean_y).sum() / (np.abs(mean_abs).sum() + 1e-9)

    return {
        "mean_absolute": mean_abs,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "areas": active_area / total_area,
        "fractional_x": float(frac_x),
        "fractional_y": float(frac_y),
    }


# ---------------------------------------------------------------------------
# Peak detection & analysis
# ---------------------------------------------------------------------------

def detect_peaks(
    raw_results: np.ndarray,
    framerate: float,
    min_peak_height: float = 0.0,
    min_peak_sep_sec: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect peaks in absolute amplitude (col 1 of raw_results)."""
    signal = raw_results[:, 1]
    min_dist = max(1, int(min_peak_sep_sec * framerate + 0.5))
    kwargs = {"distance": min_dist}
    if min_peak_height > 0:
        kwargs["height"] = min_peak_height
    indices, _ = find_peaks(signal, **kwargs)
    return raw_results[indices, 0], signal[indices]   # times, heights


def analyze_peaks(
    peak_times: np.ndarray,
    peak_heights: np.ndarray,
    skip_first: bool = True,
) -> dict:
    """Split alternating contraction/relaxation peaks and compute statistics."""
    if skip_first and len(peak_times) > 2:
        peak_times = peak_times[1:]
        peak_heights = peak_heights[1:]

    n_cycles = len(peak_times) // 2
    contract_t = peak_times[0::2][:n_cycles]
    relax_t    = peak_times[1::2][:n_cycles]
    contract_h = peak_heights[0::2][:n_cycles]
    relax_h    = peak_heights[1::2][:n_cycles]

    # contraction → relaxation interval (within cycle)
    int_time_diff = relax_t - contract_t

    # relaxation → next contraction interval (between cycles)
    n_gaps = min(len(relax_t), len(contract_t) - 1)
    relax_to_contract = (contract_t[1:n_gaps + 1] - relax_t[:n_gaps]
                         if n_gaps > 0 else np.array([]))

    # full cycle period (contraction to next contraction)
    time_diff = np.diff(contract_t)
    beat_rate = 60.0 / time_diff if len(time_diff) > 0 else np.array([])

    return {
        "contract_heights": contract_h,
        "relax_heights": relax_h,
        "contract_times": contract_t,
        "relax_times": relax_t,
        "peaks_int_time_diff": int_time_diff,         # contraction→relaxation
        "relax_to_contract_diff": relax_to_contract,  # relaxation→next contraction
        "peaks_time_diff": time_diff,
        "beat_rate_bpm": beat_rate,
        "mean_contract": float(np.mean(contract_h)) if len(contract_h) else 0.0,
        "std_contract": float(np.std(contract_h)) if len(contract_h) else 0.0,
        "mean_relax": float(np.mean(relax_h)) if len(relax_h) else 0.0,
        "std_relax": float(np.std(relax_h)) if len(relax_h) else 0.0,
        "mean_time_int": float(np.mean(int_time_diff)) if len(int_time_diff) else 0.0,
        "std_time_int": float(np.std(int_time_diff)) if len(int_time_diff) else 0.0,
        "mean_beat_rate": float(np.mean(beat_rate)) if len(beat_rate) else 0.0,
        "std_beat_rate": float(np.std(beat_rate)) if len(beat_rate) else 0.0,
    }


# ---------------------------------------------------------------------------
# Autocorrelation beat rate
# ---------------------------------------------------------------------------

def do_autocorr(
    signal: np.ndarray,
    framerate: float,
    min_beat_bpm: float = 10.0,
    max_beat_bpm: float = 300.0,
    n_peaks: int = 3,
) -> tuple[float, float]:
    """Estimate beat rate via autocorrelation — port of doAutoCorr.m.

    Returns
    -------
    beat_rate_bpm : float
    correlation   : float   peak autocorrelation value
    """
    sig = signal - signal.mean()
    corr = correlate(sig, sig, mode="full")
    corr = corr[len(corr) // 2:]
    corr /= corr[0] + 1e-9

    min_lag = int(60.0 / max_beat_bpm * framerate)
    max_lag = int(60.0 / min_beat_bpm * framerate)
    min_lag = max(1, min(min_lag, len(corr) - 1))
    max_lag = min(max_lag, len(corr) - 1)

    search = corr[min_lag:max_lag + 1]
    if len(search) == 0:
        return 0.0, 0.0

    peaks_idx, props = find_peaks(search, height=0)
    if len(peaks_idx) == 0:
        return 0.0, 0.0

    # Use mean of first n_peaks
    use = peaks_idx[:n_peaks] + min_lag
    mean_lag = float(np.mean(use))
    beat_rate = 60.0 / (mean_lag / framerate)
    correlation = float(np.mean(corr[use]))

    return beat_rate, correlation
