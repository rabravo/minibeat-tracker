# MiniBeat Tracker

Motion-based beating analysis for cardiomyocyte organoids.  
Built on [napari](https://napari.org) with numba-accelerated block matching and joblib parallelism.

## What it does

MiniBeat Tracker processes fluorescence or brightfield time-lapse images of beating cardiomyocyte organoids and extracts:

- Per-frame motion vectors via exhaustive block matching (MAD cost, JIT-compiled)
- Amplitude time series with contraction/relaxation peak detection
- Beat rate via autocorrelation and peak interval analysis
- Spatial contraction heatmaps
- Amplitude-overlay video export
- CSV exports compatible with downstream analysis

The four-step workflow mirrors the original MATLAB MotionGUI pipeline and runs entirely in a napari GUI.

## Requirements

- [Miniconda](https://docs.anaconda.com/miniconda/) or Anaconda
- [ffmpeg](https://ffmpeg.org) (for MP4 import)

## Installation

```
git clone https://github.com/rabravo/minibeat-tracker.git
cd minibeat-tracker
conda env create -f environment.yml
conda activate minibeat-tracker
```

## Running

```
conda activate minibeat-tracker
python -m minibeat-tracker
```

## Input

**Option A — TIF folder:**  
A directory of single-page grayscale TIF/TIFF files, one per frame, sorted by name.

**Option B — MP4 video (direct):**  
Select an MP4 in the Folder panel. MiniBeat extracts frames via ffmpeg into a temporary directory at the chosen FPS. Frames are deleted when the app closes.

**Option C — MP4 via ab_video_mp42tif.sh:**  
If [ab_video_mp42tif.sh](https://github.com/rabravo/usr-local-bin) is on your PATH, MiniBeat can call it with `--gray` to extract and keep TIF frames on disk alongside the video.

> TIF files must be grayscale. If your TIFs show a chroma subsampling error, re-export with `ab_video_mp42tif.sh --gray` or use the direct ffmpeg path.

## Workflow

| Step | Panel | Action |
|------|-------|--------|
| 1 | Folder | Load a TIF folder or import an MP4 |
| 2 | Vectors | Run block-matching motion estimation |
| 3 | Contraction | Compute per-frame time series and detect peaks |
| 4 | Evaluate | Analyse peak cycles and export CSVs / video |

Plots (signal, cycle analysis, contraction map) update automatically after Steps 3 and 4.

## Reference

The motion analysis workflow implemented here is based on the methodology described in:

> Huebsch, N. et al. **Automated Video-Based Analysis of Contractility and Calcium Flux in Human-Induced Pluripotent Stem Cell-Derived Cardiomyocytes Cultured over Different Spatial Scales.** *Tissue Engineering Part C: Methods* (2015). https://doi.org/10.1089/ten.tec.2014.0283

## Output

| File | Contents |
|------|----------|
| `*_BeatingData.csv` | Per-frame amplitude time series |
| `*_RawPeaks.csv` | Detected peak times and heights |
| `*_AnaPeaks.csv` | Per-cycle contraction/relaxation velocities and intervals |
| `*_AnaPeaksMean.csv` | Summary statistics across all cycles |
| `*_amplitude.mp4` | Heatmap overlay video |
