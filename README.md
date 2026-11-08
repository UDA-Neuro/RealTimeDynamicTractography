# DYNAMIC TRACTOGRAPHY

A local research viewer for Spike and HFO event sequences over FreeSurfer anatomy and tractography. The first version emphasizes a clear, interactive 3D display: a translucent cortex, turquoise fiber paths, pink Spike events, and amber HFO events. It runs in a local browser and does not upload patient data.

## Preview

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m dynamic_tractography --demo
```

The demo creates synthetic anatomy, electrodes, fibers and events. Drag to rotate, scroll to zoom, and use the timeline to replay the sequence.

## Patient data

```powershell
python -m dynamic_tractography `
  --subject 'D:\subjects\patient01' `
  --electrodes 'D:\data\electrodes.csv' `
  --spike 'D:\results\spike-run.csv' `
  --hfo 'D:\results\hfo-run.csv' `
  --tracts 'D:\data\patient-tracts.trk.gz'
```

The subject directory needs `mri/orig.mgz`, `surf/lh.pial`, and `surf/rh.pial` from FreeSurfer `recon-all`. The electrode table needs `channel_number,label,x,y,z` with coordinates in the **patient T1 scanner RAS frame, in millimetres**. For example:

```csv
channel_number,label,x,y,z
1,A1,-62.1,-18.4,25.2
2,A2,-57.8,4.9,30.5
```

Electrodes must be localized separately; a T1 reconstruction alone does not give electrode positions. Check each electrode against the T1 and verify its BESA channel number. The detector CSV files must come from the same recording and time origin. The viewer reads the existing `channel_number` and `peak_s` fields from RealTimeSpike and RealTimeHFO.

For patient tractography, export `.trk` or `.trk.gz` from DSI Studio. The viewer assumes the exported track coordinates are already in the patient T1 scanner RAS frame. If a **verified affine** maps the tract coordinates into this frame, supply its 4 × 4 matrix as text via `--tract-transform transform.txt`. An affine cannot replace a needed nonlinear registration.

The default candidate rule is: two different electrodes, same event class, peak separation from **10 to 50 ms**, and one sampled streamline passing within **8 mm** of both contacts. Adjust with `--min-delay-ms`, `--max-delay-ms`, and `--electrode-radius-mm`; use `--cross-type` to include Spike→HFO and HFO→Spike pairs. The visualized direction is defined by event order. It does not establish biological propagation or causality. Candidate paths are illustrative track segments, and streamline counts are not connection strengths.

## Current scope

- Saved event CSV files, FreeSurfer cortex, electrode table, patient `.trk`/`.trk.gz`, and local replay are implemented.
- Direct `HCP1065.fib`/`.fib.gz` import and registration are the next milestone. HCP1065 is in template space; using its fibers on a patient T1 requires validated spatial registration, generally including a nonlinear warp. Do not label unregistered template tracks as patient anatomy.
- Live watching of CSV files being appended is also planned. Current replay uses saved detector output.
- This viewer is for research visualization, not clinical diagnosis or surgical navigation.

## Test

```powershell
python -m unittest discover -s tests
```

