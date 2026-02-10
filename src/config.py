"""
HiOmicsFormer Configuration
Configure paths, device settings, and reproducibility for the project.
"""
import os
import torch
import numpy as np
import warnings

# ============================================================================
# PATHS - Modify if your data is elsewhere
# ============================================================================
# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "MLOmics", "Main_Dataset")

# Create necessary directories
SUBDIRS = [
    'checkpoints',
    'figures/main',
    'figures/supplementary',
    'results/biomarkers',
    'results/metrics',
    'results/embeddings',
    'processed_data',
    'logs'
]

for subdir in SUBDIRS:
    os.makedirs(os.path.join(PROJECT_ROOT, subdir), exist_ok=True)

# ============================================================================
# DEVICE SETUP FOR M4
# ============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple M4 GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using NVIDIA GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU (slower)")

print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"📁 Data Path: {DATA_PATH}")
print(f"✅ Data exists: {os.path.exists(DATA_PATH)}")

# ============================================================================
# SEED FOR REPRODUCIBILITY
# ============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Disable warnings
warnings.filterwarnings('ignore')

print("✅ HiOmicsFormer configuration complete!")

# Export key variables
__all__ = ['PROJECT_ROOT', 'DATA_PATH', 'device', 'SEED']
