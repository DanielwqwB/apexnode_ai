"""
SentryMesh — Full Training Runner
Run this from the project root (same folder that contains data/, model.py, etc.)

Usage:
    python run_training.py

Steps it handles:
    1. Install dependencies
    2. Copy source files to cwd if missing
    3. Run preprocessing  (data_loader.py  →  processed/)
    4. Run training loop  (train.py        →  checkpoints/)
"""

import subprocess
import sys
import os
from pathlib import Path

# ── 0. Install deps ──────────────────────────────────────────────────────────
DEPS = [
    "torch",
    "pandas",
    "numpy",
    "scikit-learn",
    "pyarrow",          # parquet support
    "fastparquet",      # fallback parquet
    # torch_scatter, torch_sparse installed via install_pyg() with the correct PyG index
]

print("=" * 60)
print("SentryMesh — Dependency Check")
print("=" * 60)

# PyG extras need special install
def install_pyg():
    """Install torch-geometric and its sparse deps."""
    import torch
    torch_ver = torch.__version__.split("+")[0]
    cuda_tag  = "cpu"
    if torch.cuda.is_available():
        cuda_tag = "cu" + torch.version.cuda.replace(".", "")

    base = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
    print(f"\nInstalling PyG extras for torch={torch_ver}, device={cuda_tag} …")
    for pkg in ["torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-f", base, "-q"],
            check=False
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch_geometric", "-q"],
        check=False
    )

def pip_install(pkg):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        check=False
    )

# Install base deps first
for dep in ["torch", "pandas", "numpy", "scikit-learn", "pyarrow"]:
    pip_install(dep)

# Then PyG
try:
    import torch_geometric
    print("torch_geometric already installed ✓")
except ImportError:
    install_pyg()

print("\nDependencies ready.\n")

# ── 1. Verify project layout ─────────────────────────────────────────────────
required_files = ["data_loader.py", "model.py", "train.py"]
missing = [f for f in required_files if not Path(f).exists()]
if missing:
    print(f"ERROR: Missing files in current directory: {missing}")
    print("Make sure data_loader.py, model.py, and train.py are in the same folder as this script.")
    sys.exit(1)

required_data = [
    "data/typhoon/train.csv",
    "data/typhoon/val.csv",
    "data/typhoon/test.csv",
    "data/gfd_qcdatabase_2019_08_01.csv",
    "data/landslide/Global_Landslide_Catalog_Export_rows.csv",
]

# Bug #4 fix: auto-extract data.zip if data folder is missing
missing_data = [f for f in required_data if not Path(f).exists()]
if missing_data:
    if Path("data.zip").exists():
        import zipfile
        print("Extracting data.zip …")
        with zipfile.ZipFile("data.zip") as z:
            z.extractall(".")
        print("Extraction complete.\n")
    # Re-check after extraction attempt
    missing_data = [f for f in required_data if not Path(f).exists()]

if missing_data:
    print(f"ERROR: Missing data files: {missing_data}")
    print("Place data.zip next to this script or extract the 'data/' folder manually.")
    sys.exit(1)

print("Project layout ✓")
print("Data files    ✓\n")

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
if not Path("processed/combined.parquet").exists():
    print("=" * 60)
    print("Step 1 / 2 — Preprocessing datasets …")
    print("=" * 60)
    result = subprocess.run([sys.executable, "data_loader.py"])
    if result.returncode != 0:
        print("\nPreprocessing failed — see errors above.")
        sys.exit(result.returncode)
    print("\nPreprocessing complete ✓\n")
else:
    print("Processed data already exists — skipping preprocessing.\n")

# ── 3. Training ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 2 / 2 — Training VigilantPath ST-GNN …")
print("=" * 60)
result = subprocess.run([sys.executable, "train.py"])
if result.returncode != 0:
    print("\nTraining failed — see errors above.")
    sys.exit(result.returncode)

print("\n" + "=" * 60)
print("✓  SentryMesh training complete!")
print("   Best model  →  checkpoints/best_model.pt")
print("   Test metrics →  checkpoints/test_metrics.json")
print("   Training log →  checkpoints/history.json")
print("=" * 60)
