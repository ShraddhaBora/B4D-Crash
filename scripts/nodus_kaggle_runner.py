"""
Before the Crash — Nodus / Kaggle Execution Notebook
=====================================================

This script is the self-contained execution entry point for Nodus Compute
or a Kaggle GPU notebook.  Copy this file (or the entire repo) to the
compute environment and run it.

Execution order:
  0. Environment detection + GPU check
  1. Dependency installation
  2. Dataset / video acquisition
  3. Model weight download
  4. Pipeline execution
  5. Experiment sweep (optional)
  6. Metric export
  7. Agent37 analysis hook

Convert to .ipynb with:
    jupytext --to notebook nodus_kaggle_runner.py

Or run directly:
    python scripts/nodus_kaggle_runner.py --video <path>
"""

# %% [markdown]
# ## 0. Environment Detection

# %%
import os, sys, platform, subprocess
from pathlib import Path

print("Python  :", sys.version)
print("Platform:", platform.system(), platform.machine())

# GPU check
try:
    import torch
    print("PyTorch :", torch.__version__)
    print("CUDA    :", torch.cuda.is_available(), "— device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
except ImportError:
    print("PyTorch not installed yet — will install below.")

# Detect environment
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IS_COLAB  = "google.colab" in sys.modules
IS_NODUS  = not IS_KAGGLE and not IS_COLAB
print(f"\nEnvironment: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Nodus/local'}")


# %% [markdown]
# ## 1. Dependency Installation

# %%
def install_deps():
    """Install pipeline dependencies on the compute node."""
    # PyTorch with CUDA (adjust cu121 for your CUDA version)
    cuda_tag = "cu121"
    torch_cmd = [
        sys.executable, "-m", "pip", "install", "-q",
        "torch", "torchvision",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
    ]
    # Core deps
    core_cmd = [
        sys.executable, "-m", "pip", "install", "-q",
        "ultralytics>=8.2",
        "opencv-python-headless>=4.9",
        "scipy>=1.12", "pandas>=2.2", "PyYAML>=6.0",
        "tqdm>=4.66", "scikit-learn>=1.4",
        "matplotlib>=3.8", "seaborn>=0.13",
    ]

    print("Installing PyTorch + CUDA...")
    subprocess.run(torch_cmd, check=False)
    print("Installing core dependencies...")
    subprocess.run(core_cmd, check=True)
    print("✓ Dependencies installed.")

# Uncomment to install:
# install_deps()


# %% [markdown]
# ## 2. Repo / Code Setup

# %%
# Target official GitHub repository (Fix #10)
GITHUB_REPO = "https://github.com/ShraddhaBora/B4D-Crash.git"

# Dynamically resolve REPO_ROOT without hardcoded workspace assumptions:
# 1. If running within an already-cloned repo directory:
if (Path.cwd() / "src" / "pipeline.py").exists():
    REPO_ROOT = Path.cwd()
elif Path("/workspace/before-the-crash").exists():
    REPO_ROOT = Path("/workspace/before-the-crash")
elif Path("/kaggle/working/B4D-Crash").exists():
    REPO_ROOT = Path("/kaggle/working/B4D-Crash")
elif Path("/kaggle/working").exists():
    REPO_ROOT = Path("/kaggle/working/B4D-Crash")
else:
    REPO_ROOT = Path("./before-the-crash").resolve()


def setup_repo():
    if REPO_ROOT.exists() and (REPO_ROOT / "src").exists():
        print(f"✓ Repo found at {REPO_ROOT}")
    else:
        print(f"Cloning repo from {GITHUB_REPO}...")
        REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", GITHUB_REPO, str(REPO_ROOT)], check=True)
        print(f"✓ Cloned to {REPO_ROOT}")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        os.chdir(REPO_ROOT)
    except Exception as e:
        print(f"Note: Could not chdir to {REPO_ROOT}: {e}")


setup_repo()


# %% [markdown]
# ## 3. Dataset / Video Acquisition

# %%
# ── Kaggle dataset download ───────────────────────────────────────────────────
# Requires kaggle.json in ~/.kaggle/
def download_kaggle_dataset(dataset_slug: str, dest: str = "data/raw"):
    """Download a Kaggle dataset. dataset_slug e.g. 'user/dataset-name'."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", dataset_slug, "-p", dest, "--unzip",
    ], check=True)
    print(f"✓ Downloaded {dataset_slug} → {dest}")

# Example (uncomment when running on Kaggle):
# download_kaggle_dataset("some-user/intersection-traffic-video", "data/raw")

# ── Manual video path (set this for Nodus) ────────────────────────────────────
VIDEO_PATH = "data/samples/sample_traffic.mp4"  # set to actual path on Nodus

# Verify video exists
if not Path(VIDEO_PATH).exists():
    print(f"⚠ Video not found at {VIDEO_PATH}")
    print("  Set VIDEO_PATH to an actual video file before running the pipeline.")
else:
    print(f"✓ Video found: {VIDEO_PATH}")


# %% [markdown]
# ## 4. Download Model Weights

# %%
def ensure_model(model_name: str = "yolov8n.pt"):
    """
    Download YOLOv8 weights if not already present.
    Ultralytics downloads them automatically on first use.
    """
    weights_dir = Path("models/checkpoints")
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / model_name
    if target.exists():
        print(f"✓ Model weights found: {target}")
        return str(target)
    # Ultralytics auto-downloads to its cache; copy to our dir
    from ultralytics import YOLO
    model = YOLO(model_name)   # downloads if not cached
    import shutil
    cache = Path.home() / ".cache" / "ultralytics" / model_name
    if cache.exists():
        shutil.copy(cache, target)
    print(f"✓ Model ready: {model_name}")
    return model_name

# MODEL_PATH = ensure_model("yolov8n.pt")   # uncomment to run
MODEL_PATH = "yolov8n.pt"


# %% [markdown]
# ## 5. Run the Pipeline

# %%
def run_pipeline(video_path: str, model: str = "yolov8n.pt",
                 experiment_config: str | None = None,
                 output_dir: str = "outputs"):
    """Execute the full Before the Crash pipeline."""
    from src.config_loader import load_config
    from src.pipeline import Pipeline

    cfg = load_config("config.yaml", experiment_config)
    cfg["video"]["input_path"] = video_path
    cfg["video"]["output_dir"] = output_dir
    cfg["detector"]["model"]   = model

    # Persist modified config for reproducibility
    import yaml, time
    ts = int(time.time())
    run_cfg_path = f"experiments/results/run_{ts}/config.yaml"
    Path(run_cfg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(run_cfg_path, "w") as fh:
        yaml.dump(cfg, fh)

    pipe = Pipeline.__new__(Pipeline)
    pipe.cfg = cfg
    from src.logger import configure_root_logger
    from src.config_loader import ensure_output_dirs
    configure_root_logger(cfg)
    ensure_output_dirs(cfg)
    pipe._lazy_init()
    result = pipe.run(video_path)

    # Save metrics
    metrics_path = f"experiments/results/run_{ts}/metrics.json"
    import json
    with open(metrics_path, "w") as fh:
        json.dump({k: v for k, v in result.items() if not isinstance(v, list)}, fh, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")

    return result

# ── Run single experiment ─────────────────────────────────────────────────────
# result = run_pipeline(VIDEO_PATH, model=MODEL_PATH)


# %% [markdown]
# ## 6. Experiment Sweep (Nodus batch mode)

# %%
def run_sweep(video_path: str, model: str, config_dir: str = "experiments/configs"):
    """
    Run all experiment configs in config_dir sequentially.
    On Nodus, parallelise with subprocess/multiprocessing as needed.
    """
    configs = sorted(Path(config_dir).glob("*.yaml"))
    print(f"Running sweep over {len(configs)} configs...")
    results = []
    for cfg_path in configs:
        print(f"\n{'─'*50}")
        print(f"Experiment: {cfg_path.name}")
        r = run_pipeline(video_path, model=model,
                         experiment_config=str(cfg_path),
                         output_dir=f"outputs/sweep/{cfg_path.stem}")
        results.append({"config": cfg_path.name, **{k: v for k, v in r.items() if not isinstance(v, list)}})
    return results

# results = run_sweep(VIDEO_PATH, MODEL_PATH)


# %% [markdown]
# ## 7. Agent37 Analysis Hook

# %%
def run_agent37_analysis(results_dir: str = "experiments/results"):
    """Invoke Agent37 analysis after sweep completes."""
    subprocess.run([
        sys.executable, "scripts/agent37_analyze.py",
        "--results-dir", results_dir,
        "--generate-next",
    ], check=False)

# run_agent37_analysis()


# %% [markdown]
# ## 8. Inspect Outputs

# %%
def inspect_outputs(output_dir: str = "outputs"):
    """Print a summary of all generated output files."""
    root = Path(output_dir)
    for f in sorted(root.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  {f.relative_to(root):<55} {size_kb:7.1f} KB")

# inspect_outputs()


# %% [markdown]
# ## Quick-start one-liner
# Uncomment the cell below for a full end-to-end run:

# %%
# ── FULL RUN (one command) ────────────────────────────────────────────────────
# Adjust VIDEO_PATH before executing.
#
# install_deps()
# setup_repo()
# result = run_pipeline(VIDEO_PATH, model="yolov8n.pt")
# inspect_outputs()
# ─────────────────────────────────────────────────────────────────────────────
print("Nodus/Kaggle runner loaded. Set VIDEO_PATH and uncomment cells to execute.")
