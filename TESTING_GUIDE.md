# Before the Crash — Testing & Platform Validation Guide

This guide provides step-by-step instructions to validate the complete **Before the Crash** near-miss conflict detection pipeline across:
1. **Local Machine (Windows)** — Unit tests, dry-run simulation, and offline safety module validation (no GPU required).
2. **Nodus Compute (Dedicated GPU)** — Official execution environment for real-video ML inference.
3. **Kaggle GPU Notebooks (Secondary GPU Fallback)** — Cloud GPU execution path with T4/P100 accelerators.

---

## Quick Reference Summary

| Environment | Purpose | GPU Required? | Commands |
|---|---|:---:|---|
| **Local Machine** | Dev, linting, tests, dry-run | ❌ No | `python -m pytest tests/ -v`<br>`python scripts/dry_run.py` |
| **Nodus Compute** | Production ML pipeline on real video | ✅ Yes (CUDA) | `python scripts/nodus_kaggle_runner.py --video <path>` |
| **Kaggle Notebook** | Cloud GPU execution fallback | ✅ Yes (CUDA) | `jupytext --to notebook scripts/nodus_kaggle_runner.py`<br>or direct python execution |

---

## Part 1: Local Machine Testing (Windows)

Use your local machine for rapid verification, unit testing, and lightweight mock validation.

### Step 1.0 — Verify Environment & Dependencies
```powershell
cd C:\Users\works\Desktop\malkist\before-the-crash
python --version
# Expected: Python 3.10+ (e.g., Python 3.13.x)

python -c "import numpy, scipy, pandas, yaml, cv2; print('[OK] Core dependencies available')"
```

---

### Step 1.1 — Run Deterministic Test Suite (26 Tests)
Run all deterministic tests covering the corrected TTC conflict radius, temporal sampling across interaction windows, event merging, conflict type compatibility, and motorcycle classification:

```powershell
python -m pytest tests/ -v
```

**Expected output:**
```text
============================= 26 passed in ~3-5s ==============================
tests/test_audit_fixes.py::test_ttc_closest_approach_conflict_radius PASSED
tests/test_audit_fixes.py::test_multi_timestamp_ttc PASSED
tests/test_audit_fixes.py::test_event_manager_temporal_merging PASSED
tests/test_audit_fixes.py::test_evasive_action_proximity PASSED
tests/test_audit_fixes.py::test_pet_non_crossing_inapplicable PASSED
tests/test_audit_fixes.py::test_sustained_approach_check PASSED
tests/test_audit_fixes.py::test_evaluator_conflict_type_compatibility PASSED
tests/test_audit_fixes.py::test_agent37_dynamic_config PASSED
tests/test_audit_fixes.py::test_motorcycle_classification PASSED
tests/test_audit_fixes.py::test_pipeline_temporal_sampling_integration PASSED
tests/test_audit_fixes.py::test_runner_cuda_fail_fast PASSED
tests/test_pipeline.py::TestTTC::... PASSED
...
```

---

### Step 1.2 — Run Pipeline Dry-Run Simulation
Simulate a complete 10-second traffic scene end-to-end without needing a GPU or real video:

```powershell
python scripts/dry_run.py --frames 150 --tracks 6
```

**What this exercises:**
- Synthetic video generation (`outputs/dry_run/synthetic.mp4`)
- Frame ingestion & mock detections
- Trajectory reconstruction & Savitzky-Golay smoothing
- Spatial zone grid assignment (5x5 grid)
- Candidate interaction pairing with sustained approach filtering
- Multi-sample temporal safety analysis (TTC & PET over time)
- Temporal event merging via `EventManager`
- Evidence clip extraction & annotated video rendering (`outputs/dry_run/annotated_video.mp4`)
- Spatial hotspot ranking report

---

### Step 1.3 — Run Agent37 AI Safety Reasoning Layer
Validate the Agent37 reasoning layer against deterministic dummy risk output or pipeline CSVs:

```powershell
# 1. Run all Agent37 unit tests (9 tests)
python -m pytest tests/test_agent37.py -v

# 2. Run Agent37 demonstration on dummy risk output
python scripts/demo_agent37.py --input data/dummy_risk_output.json

# 3. Run Agent37 directly against pipeline dry-run CSV outputs
python scripts/demo_agent37.py --events-csv outputs/dry_run/events/conflict_events.csv --hotspots-csv outputs/dry_run/reports/hotspot_summary.csv
```

**What this exercises:**
- Pydantic v2 data contracts (`RiskAnalysisInput` and `AgentAnalysisOutput`)
- Authoritative ML risk score preservation (verifies ML scores are unaltered)
- Strict evidence grounding (every finding, pattern, hotspot, and recommendation cites valid event IDs)
- Cautious assessment language verification
- Honest uncertainty and limitation statements
- Machine-readable structured output export to `outputs/reports/agent37_safety_intelligence.json`

---

### Step 1.4 — Verify Fail-Fast CUDA Protection Locally
Confirm that the execution runner enforces the GPU constraint and refuses accidental CPU execution:

```powershell
# Expect failure on CPU machines without GPU accelerator:
python scripts/nodus_kaggle_runner.py --check-gpu
```

**Expected output:**
```text
RuntimeError: CUDA GPU accelerator is required for Nodus / Kaggle execution. Accidental CPU execution for this heavy ML workload is strictly prohibited.
```

---

## Part 2: Platform 1 — Nodus Compute Execution

**Nodus Compute** is the primary environment for real-video ML inference.

### Step 2.1 — Access Nodus Environment & Clone Codebase
Log in to your Nodus Compute instance and clone the official repository:

```bash
git clone https://github.com/ShraddhaBora/B4D-Crash.git
cd B4D-Crash
```

### Step 2.2 — Install Compute Dependencies
Ensure PyTorch with CUDA support and computer vision dependencies are present:

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core pipeline dependencies
pip install ultralytics>=8.2 opencv-python-headless>=4.9 scipy>=1.12 pandas>=2.2 PyYAML>=6.0 tqdm>=4.66 scikit-learn>=1.4 matplotlib>=3.8 seaborn>=0.13
```

### Step 2.3 — Verify CUDA Acceleration
```bash
python scripts/nodus_kaggle_runner.py --check-gpu
```

**Expected output:**
```text
[OK] CUDA accelerator confirmed: NVIDIA A100 / RTX / Tesla ... (CUDA 12.x)
```

### Step 2.4 — Execute Real-Video Inference
Run the end-to-end pipeline on your real traffic video:

```bash
python scripts/nodus_kaggle_runner.py \
    --video /path/to/intersection_video.mp4 \
    --model yolov8n.pt \
    --output-dir outputs/nodus_run
```

**Pipeline Stages Executed:**
1. **Ingestion**: Decodes video stream at native frame rate.
2. **Perception**: GPU-accelerated YOLOv8 detection + ByteTrack tracking.
3. **Trajectory Reconstruction**: Position smoothing and velocity/acceleration estimation.
4. **Candidate Generation**: Sustained closing approach detection.
5. **Safety Metrics**: Multi-timestamp closest-approach TTC, conflict radius thresholding, crossing PET, evasive action windowing.
6. **Event Generation**: Temporal clustering and merging within `merge_window`.
7. **Evidence & Visualization**: MP4 clip extraction around peak conflicts, annotated full video, CSV event logs.
8. **Hotspot Analysis**: Spatial concentration ranking across zones.

### Step 2.5 — Run Parameter Sensitivity Sweep
To evaluate multiple parameter profiles (Conservative, Moderate, Sensitive):

```bash
python scripts/nodus_kaggle_runner.py \
    --video /path/to/intersection_video.mp4 \
    --sweep
```

### Step 2.6 — Trigger Agent37 Autonomous Optimization
Analyze sweep results and generate the empirical next-iteration configuration:

```bash
python scripts/agent37_analyze.py --results-dir experiments/results/ --generate-next
```
Agent37 writes the tuned YAML to `experiments/configs/exp_next_agent37.yaml`.

---

## Part 3: Platform 2 — Kaggle GPU Notebook Execution

Use Kaggle GPU instances (T4 x2 or P100) as the secondary fallback environment.

### Step 3.1 — Create Notebook & Enable Accelerator
1. Create a new notebook on Kaggle.
2. In the right-hand panel: **Notebook Options** $\rightarrow$ **Accelerator** $\rightarrow$ Select **GPU T4** or **GPU P100**.
3. Under **Internet**, toggle **Internet on**.

### Step 3.2 — Clone Repository
In the first code cell:

```python
!git clone https://github.com/ShraddhaBora/B4D-Crash.git /kaggle/working/B4D-Crash
%cd /kaggle/working/B4D-Crash
```

### Step 3.3 — Install Dependencies & Verify GPU
```python
!pip install -q ultralytics opencv-python-headless
!python scripts/nodus_kaggle_runner.py --check-gpu
```

### Step 3.4 — Mount Traffic Dataset
- Option A (Kaggle Dataset): Add dataset via the UI (`Add Data`) or run:
  ```python
  # Example: download public traffic video
  # !kaggle datasets download -d <user/dataset> -p data/raw --unzip
  ```
- Option B (Sample or Direct URL):
  ```python
  !wget -O data/samples/traffic.mp4 "https://example.com/traffic_video.mp4"
  ```

### Step 3.5 — Execute Pipeline
```python
!python scripts/nodus_kaggle_runner.py \
    --video data/samples/traffic.mp4 \
    --model yolov8n.pt \
    --output-dir /kaggle/working/outputs
```

### Step 3.6 — Inspect & Download Artifacts
```python
import os
for root, dirs, files in os.walk("/kaggle/working/outputs"):
    for file in files:
        path = os.path.join(root, file)
        size_kb = os.path.getsize(path) / 1024
        print(f"{os.path.relpath(path, '/kaggle/working/outputs'):<50} {size_kb:7.1f} KB")
```

---

## Output Artifacts Checklist

Every full pipeline run produces the following artifacts:

| Output File | Format | Contents |
|---|---|---|
| `events/conflict_events.csv` | CSV | Event ID, road user classes, conflict type (`VEH_PED`, `VEH_MOTO`, `VEH_VEH`), min TTC, min PET, peak conflict timestamp & frame, max speed change, severity, risk score. |
| `clips/event_<id>.mp4` | MP4 | Video clips bounded around the peak conflict timestamp for human review. |
| `annotated_video.mp4` | MP4 | Full video rendered with bounding boxes, tracking IDs, trajectory trails, and active conflict alerts. |
| `reports/hotspot_summary.csv` | CSV | Spatial zone risk rankings, incident frequency, and road user distributions. |
| `metrics.json` | JSON | Summary execution metrics, frame rate, total events, and classification breakdown. |

---

## Troubleshooting & Verification Checklist

- **Issue**: `RuntimeError: CUDA GPU accelerator is required...`
  - **Fix**: Ensure you have selected a GPU runtime (on Kaggle: Accelerator $\rightarrow$ GPU T4; on Nodus: ensure `nvidia-smi` shows active GPU devices).
- **Issue**: `Video file not found`
  - **Fix**: Check video file path with `ls -l <path>` and pass `--video /absolute/path/to/video.mp4`.
- **Issue**: `ModuleNotFoundError: No module named 'src'`
  - **Fix**: Ensure your working directory is the repository root (`cd /path/to/B4D-Crash`).
