# Before the Crash — Manual Testing Guide

Step-by-step instructions to validate everything we've built, in order,
on your local Windows machine. No GPU needed for any step until Step 5.

---

## Prerequisites

Make sure you are in the project folder for every command:

```powershell
cd C:\Users\works\Desktop\malkist\before-the-crash
```

---

## Step 0 — Verify Python + Dependencies

```powershell
python --version
# Expected: Python 3.13.x

python -c "import numpy, scipy, pandas, yaml, cv2; print('Core deps OK')"
# Expected: Core deps OK
```

---

## Step 1 — Run Unit Tests (15 deterministic tests, no GPU)

```powershell
python -m pytest tests/test_pipeline.py -v
```

**Expected output:**
```
15 passed in ~1-2s
```

Tests cover: TTC, PET, smoothing, conflict classifier, risk score, event manager, zones.

---

## Step 2 — Run the Pipeline Dry-Run (no GPU, no real video, no ultralytics)

This generates a synthetic video, runs mock detection + tracking, then exercises
the full pipeline: trajectories → TTC/PET → events → evidence clips → hotspots.

```powershell
$env:PYTHONIOENCODING = "utf-8"
python scripts/dry_run.py --frames 150 --tracks 6
```

**Expected output (summary at the end):**
```
DRY RUN COMPLETE
  Frames processed:       150
  Detections:             ~700-900
  Trajectories (GOOD):    3-6
  Candidate interactions: 1+
  Conflict events:        1+
  Evidence clips:         1-3
  Hotspot zones:          1+
  Wall-clock time:        < 30s
```

**Outputs generated in `outputs/dry_run/`:**
- `detections/detections.csv`
- `tracks/tracks.csv`
- `events/conflict_events.csv`
- `clips/event_*.mp4`
- `annotated_video.mp4`  ← open this to visually verify bboxes + trails
- `reports/hotspot_summary.csv`

---

## Step 3 — Inspect the Dry-Run Outputs

### 3a. Check detections CSV
```powershell
python -c "
import pandas as pd
df = pd.read_csv('outputs/dry_run/detections/detections.csv')
print(df.head())
print('Columns:', list(df.columns))
print('Rows:', len(df))
print('Classes:', df.class_name.value_counts().to_dict())
"
```

### 3b. Check tracks CSV
```powershell
python -c "
import pandas as pd
df = pd.read_csv('outputs/dry_run/tracks/tracks.csv')
print('Unique track IDs:', sorted(df.track_id.unique()))
print('Frames per track:')
print(df.groupby('track_id').size())
"
```

### 3c. Check conflict events
```powershell
python -c "
import pandas as pd
df = pd.read_csv('outputs/dry_run/events/conflict_events.csv')
if len(df) == 0:
    print('No events — check candidate_pairs threshold (max_candidate_distance in config.yaml)')
else:
    print(df[['event_id','conflict_type','zone_id','severity','risk_score','reasons']])
"
```

### 3d. Check hotspot summary
```powershell
python -c "
import pandas as pd
df = pd.read_csv('outputs/dry_run/reports/hotspot_summary.csv')
print(df)
"
```

### 3e. Play the annotated video
```powershell
start outputs\dry_run\annotated_video.mp4
```
You should see coloured bounding boxes with track IDs, trajectory trails,
and a red banner for any active conflict events.

---

## Step 4 — Test Config Loading + Deep Merge

```powershell
python -c "
from src.config_loader import load_config
# Base config
cfg = load_config('config.yaml')
print('Base TTC threshold:', cfg['safety']['ttc_threshold'])

# Overlay experiment config
cfg2 = load_config('config.yaml', 'experiments/configs/exp_001_conservative.yaml')
print('Conservative TTC threshold:', cfg2['safety']['ttc_threshold'])
# Should be 2.0 (from the overlay)
"
```

---

## Step 5 — Test Individual Safety Modules (no video needed)

### TTC — head-on approach
```powershell
python -c "
import sys; sys.path.insert(0, '.')
import numpy as np, math
from src.schema import Trajectory, TrackQuality
from src.trajectory.smoothing import smooth_positions
from src.trajectory.motion import compute_motion
from src.safety.ttc import compute_ttc
from src.schema import CandidateInteraction

# Car A moving right at 50px/s, Car B moving left from x=200
def make_traj(tid, cls, positions):
    ts = np.linspace(0, len(positions)/10, len(positions))
    raw = np.array(positions, float)
    sm = smooth_positions(raw, 3)
    vels, spd, hdg, acc = compute_motion(sm, ts)
    return Trajectory(tid, cls, TrackQuality.GOOD, ts, raw, sm, vels, spd, hdg, acc)

ta = make_traj(1, 'car', [[i*5, 100] for i in range(20)])
tb = make_traj(2, 'car', [[200-i*5, 100] for i in range(20)])
cand = CandidateInteraction(1,2,'car','car','VEH_VEH', 0,2,0,20)
result = compute_ttc(ta, tb, cand, t_query=0.5)
print('TTC (head-on):', result.value, '  valid:', result.valid, '  model:', result.model)
assert result.valid and result.value < 10, 'TTC test FAILED'
print('TTC test PASSED')
"
```

### Risk Score — ensure components add up correctly
```powershell
python -c "
from src.schema import CandidateInteraction, TTCResult, PETResult
from src.safety.risk_score import compute_risk_score
import math

cand = CandidateInteraction(1,2,'car','person','VEH_PED',0,3,0,90)
ttc  = TTCResult(0.5, True, 'closest_approach', 0.9)
pet  = PETResult(0.3, True, (100,100), 0.9)
risk = compute_risk_score(cand, ttc, pet, [], ttc_threshold=3.0, pet_threshold=2.0)
print('TTC score:    ', risk.ttc_score)
print('PET score:    ', risk.pet_score)
print('Vuln score:   ', risk.vulnerability_score)
print('Overall:      ', risk.overall_score)
assert risk.overall_score > 0.3, 'Risk score unexpectedly low'
print('Risk score test PASSED')
"
```

---

## Step 6 — Test Video Ingestion on a Real File

If you have any `.mp4` file handy, test ingestion:

```powershell
python -c "
from src.ingestion.video_reader import get_video_metadata, VideoReader
import sys

vid = sys.argv[1] if len(sys.argv) > 1 else 'outputs/dry_run/synthetic.mp4'
meta = get_video_metadata(vid)
print('FPS:', meta.fps)
print('Frames:', meta.frame_count)
print('Resolution:', meta.width, 'x', meta.height)
print('Duration:', round(meta.duration_s, 2), 's')

# Read first 5 frames
count = 0
for frame, fid, ts, fps in VideoReader(vid, frame_skip=1):
    print(f'  frame {fid} @ t={ts:.3f}s  shape={frame.shape}')
    count += 1
    if count >= 5:
        break
print('Ingestion OK')
" outputs/dry_run/synthetic.mp4
```

---

## Step 7 — Experiment Config Validation

```powershell
python -c "
from src.config_loader import load_config
import yaml

for name in ['exp_001_conservative', 'exp_002_moderate', 'exp_003_sensitive']:
    cfg = load_config('config.yaml', f'experiments/configs/{name}.yaml')
    print(f'{name}:')
    print(f'  TTC threshold:  {cfg[\"safety\"][\"ttc_threshold\"]}')
    print(f'  PET threshold:  {cfg[\"safety\"][\"pet_threshold\"]}')
    print(f'  Merge window:   {cfg[\"events\"][\"merge_window\"]}')
"
```

---

## Step 8 — (Nodus/Kaggle only) Run Full Pipeline on Real Video

Once you have a real traffic video on Nodus or Kaggle:

```bash
# Install deps on compute node
pip install ultralytics>=8.2 opencv-python-headless scipy pandas PyYAML tqdm scikit-learn

# Run the pipeline
python scripts/run_pipeline.py \
    --video /path/to/traffic_video.mp4 \
    --config config.yaml \
    --experiment experiments/configs/exp_002_moderate.yaml
```

Or use the self-contained notebook runner:

```bash
python scripts/nodus_kaggle_runner.py
# (edit VIDEO_PATH and uncomment the final cell first)
```

---

## Step 9 — Agent37 Analysis (after running experiments on Nodus)

After running multiple experiment configs on Nodus and saving results to `experiments/results/`:

```powershell
python scripts/agent37_analyze.py --results-dir experiments/results/ --generate-next
```

**Expected output:**
- Comparison table across experiments (Precision / Recall / F1 / false alerts/min)
- Best-performing config identification
- Threshold sensitivity analysis
- New recommended config written to `experiments/configs/exp_next_agent37.yaml`

---

## Checkpoint: What Passes = Pipeline Is Ready for Real Video

| Check | Command | Pass condition |
|-------|---------|----------------|
| Core deps | `python -c "import numpy, scipy, cv2"` | No error |
| Unit tests | `pytest tests/ -v` | 15/15 passed |
| Dry run | `python scripts/dry_run.py` | Completes, outputs generated |
| Annotated video | `start outputs\dry_run\annotated_video.mp4` | Boxes + trails visible |
| Config merge | Step 4 above | Conservative TTC = 2.0 |
| TTC module | Step 5 above | TTC test PASSED |
| Risk score | Step 5 above | Risk score test PASSED |

Once all boxes are checked, the pipeline is confirmed end-to-end on synthetic data
and is ready to be pointed at real video on Nodus Compute.

---

## Troubleshooting

**`UnicodeEncodeError` with checkmarks on Windows terminal:**
```powershell
$env:PYTHONIOENCODING = "utf-8"
# Then re-run the command
```

**`ModuleNotFoundError: ultralytics`:**  
This only affects `scripts/run_pipeline.py` (the real pipeline with YOLO).  
The dry-run and unit tests do NOT require ultralytics.

**`No conflict events found` in dry-run:**  
Increase `interaction.max_candidate_distance` in `config.yaml` (try 200-300) or
decrease `trajectory.min_track_length` to 5.

**Tracker produces very few GOOD trajectories:**  
Reduce `trajectory.min_track_length` in `config.yaml`.
