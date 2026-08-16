"""Vector cleaning — port of CleanMotionVectors.m, FFTFilterBasedOnVectorComp.m,
CleanBasedOnVariation.m."""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, irfft


def clean_neighbor(
    motion_vect: np.ndarray,
    motion_amp: np.ndarray,
    threshold: float = 2.0,
    use_amplitude: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Neighbor-based outlier removal.

    Blocks whose amplitude (or vector magnitude) deviates more than
    `threshold` × local neighbourhood std are zeroed.
    """
    vect = motion_vect.copy()
    amp = motion_amp.copy()

    field = amp if use_amplitude else np.linalg.norm(vect, axis=2)  # [R,C,T]

    R, C, T = field.shape
    for t in range(T):
        frame = field[:, :, t]
        for r in range(R):
            for c in range(C):
                neighbours = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C and not (dr == 0 and dc == 0):
                            neighbours.append(frame[nr, nc])
                if neighbours:
                    nb = np.array(neighbours)
                    if abs(frame[r, c] - nb.mean()) > threshold * nb.std() + 1e-9:
                        vect[r, c, :, t] = 0.0
                        amp[r, c, t] = 0.0

    return vect, amp


def clean_fft(
    motion_vect: np.ndarray,
    framerate: float,
    freq_thresh: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """FFT-based temporal filter on each block's vector components.

    Components with frequency > freq_thresh × fundamental are attenuated.
    Mirrors FFTFilterBasedOnVectorComp.m.
    """
    vect = motion_vect.copy()
    R, C, _, T = vect.shape

    fundamental = framerate / T
    cutoff_bin = int(freq_thresh / fundamental) + 1

    for r in range(R):
        for c in range(C):
            for component in range(2):
                signal = vect[r, c, component, :]
                spectrum = rfft(signal)
                spectrum[cutoff_bin:] = 0.0
                vect[r, c, component, :] = irfft(spectrum, n=T)

    amp = np.linalg.norm(vect, axis=2)  # [R, C, T]
    return vect, amp


def clean_variation(
    motion_vect: np.ndarray,
    motion_amp: np.ndarray,
    cov_threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Variation-based cleaning: remove blocks with high CoV over time.

    A block whose temporal coefficient of variation exceeds `cov_threshold`
    is considered noise and zeroed. Mirrors CleanBasedOnVariation.m.
    """
    vect = motion_vect.copy()
    amp = motion_amp.copy()

    # CoV per block: std / (mean + eps)
    mean_amp = amp.mean(axis=2)          # [R, C]
    std_amp = amp.std(axis=2)            # [R, C]
    cov = std_amp / (mean_amp + 1e-9)   # [R, C]

    noisy = cov > cov_threshold
    vect[noisy, :, :] = 0.0
    amp[noisy, :] = 0.0

    return vect, amp
