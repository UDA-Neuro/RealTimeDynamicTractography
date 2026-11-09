# DYNAMIC TRACTOGRAPHY

## Install (Windows PowerShell)

```powershell
py -3.12 -m venv "$HOME\.venv"
& "$HOME\.venv\Scripts\python.exe" -m pip install --upgrade "https://github.com/UDA-Neuro/RealTimeDynamicTractography/archive/refs/heads/main.zip"
```

## Run the demo

```powershell
& "$HOME\.venv\Scripts\python.exe" -m dynamic_tractography --demo
```

If the atlas is not cached, the first run downloads the HCP1065 TRK atlas (about 588 MB). To use an existing download:

```powershell
& "$HOME\.venv\Scripts\python.exe" -m dynamic_tractography --demo --demo-atlas 'D:\data\hcp1065_avg_tracts_trk.zip'
```

Atlas: [Yeh 2022 HCP1065](https://brain.labsolver.org/hcp_trk_atlas.html), derived from WU-Minn HCP data.

Select a cluster, press Play, or use **Save Cluster Video** to download its WebM clip.

Drag to rotate, scroll to zoom, and use the timeline to replay.

## Run with your data

```powershell
& "$HOME\.venv\Scripts\python.exe" -m dynamic_tractography `
  --subject 'D:\subjects\patient01' `
  --electrodes 'D:\data\electrodes.csv' `
  --spike 'D:\results\spike-run.csv' `
  --hfo 'D:\results\hfo-run.csv' `
  --tracts 'D:\data\patient-tracts.trk.gz'
```

Inputs:

- `--subject`: FreeSurfer subject directory containing `mri/orig.mgz`, `surf/lh.pial`, and `surf/rh.pial`.
- `--electrodes`: CSV with `channel_number,label,x,y,z`. Coordinates are in the T1 scanner RAS frame, in millimetres.
- `--spike` / `--hfo`: CSV output from RealTimeSpike / RealTimeHFO.
- `--tracts`: DSI Studio exported `.trk` or `.trk.gz` in the T1 scanner RAS frame. If an affine transform is needed, pass a 4 × 4 text matrix using `--tract-transform`.

Example electrode CSV:

```csv
channel_number,label,x,y,z
1,A1,-62.1,-18.4,25.2
2,A2,-57.8,4.9,30.5
```

Optional parameters: `--min-delay-ms` (default 10), `--max-delay-ms` (default 50), `--electrode-radius-mm` (default 8), and `--cross-type`.


