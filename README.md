# Before the Crash

Video-based traffic conflict analysis pipeline.  
Detects potential near-miss interactions between road users **before** they become recorded crashes.

---

## What it does

```
Traffic Video
    ↓
YOLO Detection  (YOLOv8, pretrained)
    ↓
ByteTrack       (persistent multi-object tracking)
    ↓
Trajectory Reconstruction + Smoothing
    ↓
Candidate Interaction Generation
    ↓
TTC / PET Analysis
    ↓
Evasive Action Detection
    ↓
Rule-Based Conflict Classification
    ↓
Risk Scoring
    ↓
Event Merging
    ↓
Evidence Clips + Annotated Video
    ↓
Hotspot Aggregation + Ranking
```

---

## Quick Start (local inspection only — no GPU required)

```bash
# 1. Install (pure-CPU for local dev)
pip install -r requirements.txt

# 2. Run unit tests
cd before-the-crash
python -m pytest tests/ -v
```

---

## Running the Pipeline on Nodus or Kaggle

**The ML workload runs exclusively on Nodus Compute (preferred) or a Kaggle GPU notebook.**

```bash
# On Nodus / Kaggle — run the self-contained runner:
python scripts/nodus_kaggle_runner.py --video data/samples/clip.mp4
```

Or step-by-step via the pipeline CLI:

```bash
python scripts/run_pipeline.py \
    --video data/samples/clip.mp4 \
    --config config.yaml \
    --experiment experiments/configs/exp_002_moderate.yaml
```

---

## Experiment Sweep + Agent37 Analysis

```bash
# Run three threshold configurations on Nodus
python scripts/run_pipeline.py --video clip.mp4 --experiment experiments/configs/exp_001_conservative.yaml
python scripts/run_pipeline.py --video clip.mp4 --experiment experiments/configs/exp_002_moderate.yaml
python scripts/run_pipeline.py --video clip.mp4 --experiment experiments/configs/exp_003_sensitive.yaml

# Agent37 compares results and recommends the next config
python scripts/agent37_analyze.py --results-dir experiments/results/ --generate-next
```

---

## Project Structure

```
before-the-crash/
├── config.yaml                    ← all thresholds here, none hard-coded
├── requirements.txt
├── src/
│   ├── schema.py                  ← shared typed data classes
│   ├── logger.py
│   ├── config_loader.py
│   ├── pipeline.py                ← main orchestrator
│   ├── ingestion/video_reader.py
│   ├── perception/
│   │   ├── detector.py            ← YOLOv8 wrapper
│   │   └── tracker.py             ← ByteTrack wrapper
│   ├── trajectory/
│   │   ├── builder.py             ← quality checks + reconstruction
│   │   ├── smoothing.py           ← Savitzky-Golay
│   │   └── motion.py              ← velocity / heading / acceleration
│   ├── geometry/
│   │   ├── calibration.py         ← optional homography
│   │   └── zones.py               ← spatial zones
│   ├── safety/
│   │   ├── candidate_pairs.py     ← proximity + approach filter
│   │   ├── ttc.py                 ← Time-To-Collision
│   │   ├── pet.py                 ← Post-Encroachment Time
│   │   ├── evasive_action.py      ← braking / swerve / stop detection
│   │   ├── conflict_classifier.py ← rule-based (interpretable)
│   │   └── risk_score.py          ← component-based score
│   ├── events/
│   │   ├── event_manager.py       ← merge repeated alerts
│   │   └── evidence.py            ← clips + annotated video
│   ├── hotspots/
│   │   ├── aggregation.py
│   │   └── ranking.py
│   └── evaluation/
│       └── evaluator.py           ← TP/FP/FN / P / R / F1
├── scripts/
│   ├── run_pipeline.py            ← CLI entry point
│   ├── nodus_kaggle_runner.py     ← Nodus/Kaggle execution notebook
│   └── agent37_analyze.py         ← Agent37 experiment analysis
├── experiments/
│   └── configs/
│       ├── exp_001_conservative.yaml
│       ├── exp_002_moderate.yaml
│       └── exp_003_sensitive.yaml
└── tests/
    └── test_pipeline.py           ← 15 deterministic unit tests
```

---

## Compute Architecture

| Layer | Machine | Role |
|-------|---------|------|
| Development | Local Windows | Edit code, inspect outputs, run tests |
| GPU Inference | **Nodus Compute** (preferred) | YOLO inference, tracking, full pipeline |
| GPU Fallback  | **Kaggle GPU Notebook** | Same pipeline, Kaggle dataset access |
| Analysis | Agent37 | Compare experiment results, recommend next config |

---

## Configuration

All thresholds are in `config.yaml`.  No magic numbers in source code.

Key parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `safety.ttc_threshold` | 3.0 s | Below this → conflict candidate |
| `safety.pet_threshold` | 2.0 s | Below this → conflict candidate |
| `safety.speed_change_threshold` | 0.35 | Fraction of speed drop = evasive braking |
| `interaction.max_candidate_distance` | 150 px | Spatial filter for pairs |
| `events.merge_window` | 2.0 s | Merge nearby repeated alerts |

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/detections/detections.csv` | All YOLO detections |
| `outputs/tracks/tracks.csv` | ByteTrack observations |
| `outputs/trajectories/trajectories.csv` | Smoothed + motion-estimated trajectories |
| `outputs/events/conflict_events.csv` | Merged conflict events with metrics |
| `outputs/clips/event_*.mp4` | Annotated evidence clips |
| `outputs/annotated_video.mp4` | Full annotated output |
| `outputs/reports/hotspot_summary.csv` | Per-zone risk ranking |
| `outputs/reports/pipeline_summary.json` | Run summary |

---

## Disclaimer

This system detects and analyzes **potential traffic conflicts** from video.  
It does **not** predict crash probability or guarantee that any detected event is a confirmed near-miss.  
All risk scores are prototype indicators requiring validation against ground-truth data before operational use.
