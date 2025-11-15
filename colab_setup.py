"""
Google Colab Setup Script for T4 GPU Experiments
=================================================

Instructions:
1. Open Google Colab: https://colab.research.google.com/
2. Create a new notebook
3. **IMPORTANT: Enable GPU FIRST!**
   - Click: Runtime > Change runtime type
   - Hardware accelerator: GPU (T4)
   - Click: Save
4. Copy and run the cells below

This script will:
- Verify T4 GPU is enabled
- Clone your GitHub repository
- Download ImageNette dataset
- Run experiments on T4 GPU
- Save results to download
"""

# ============================================================================
# VERIFY GPU IS ENABLED (CRITICAL!)
# ============================================================================
print("=" * 80)
print("CHECKING GPU AVAILABILITY")
print("=" * 80)

import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU: {gpu_name}")
    print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    print("\n✓ GPU IS ENABLED - Ready to proceed!")
else:
    print("\n" + "=" * 80)
    print("❌ ERROR: NO GPU DETECTED!")
    print("=" * 80)
    print("\nYou MUST enable GPU before running experiments:")
    print("1. Click: Runtime > Change runtime type")
    print("2. Hardware accelerator: Select 'GPU' or 'T4 GPU'")
    print("3. Click: Save")
    print("4. Re-run this cell to verify")
    print("\n" + "=" * 80)
    sys.exit("STOPPING: GPU not enabled. Enable GPU and try again.")

# ============================================================================
# Clone Repository and Download Dataset
# ============================================================================
print("=" * 80)
print("CLONING REPOSITORY")
print("=" * 80)

# Clone your repository
!git clone https://github.com/dhairyamishra/cloud-training-and-analysis-nyu25.git
%cd cloud-training-and-analysis-nyu25

# Download ImageNette dataset
print("\n" + "=" * 80)
print("DOWNLOADING IMAGENETTE DATASET")
print("=" * 80)
!python download_imagenette.py

# ============================================================================
# Create Results Directory
# ============================================================================
import os
os.makedirs('results/envC_T4', exist_ok=True)

# ============================================================================
# Run ResNet18 Experiment
# ============================================================================
print("=" * 80)
print("EXPERIMENT 1: ResNet18, Batch 128")
print("=" * 80)

!python scripts/main_modified.py data/imagenette2-320 \
    -a resnet18 \
    -b 128 \
    --epochs 1 \
    --max-iters 110 \
    --warmup-iters 10 \
    --log-file results/envC_T4/resnet18_b128.csv \
    --gpu 0

# ============================================================================
# Run ResNet50 Experiments (Batch Sweep)
# ============================================================================
print("=" * 80)
print("EXPERIMENT 2: ResNet50, Batch 32")
print("=" * 80)

!python scripts/main_modified.py data/imagenette2-320 \
    -a resnet50 \
    -b 32 \
    --epochs 1 \
    --max-iters 110 \
    --warmup-iters 10 \
    --log-file results/envC_T4/resnet50_b32.csv \
    --gpu 0

print("\n" + "=" * 80)
print("EXPERIMENT 3: ResNet50, Batch 64")
print("=" * 80)

!python scripts/main_modified.py data/imagenette2-320 \
    -a resnet50 \
    -b 64 \
    --epochs 1 \
    --max-iters 110 \
    --warmup-iters 10 \
    --log-file results/envC_T4/resnet50_b64.csv \
    --gpu 0

print("\n" + "=" * 80)
print("EXPERIMENT 4: ResNet50, Batch 128")
print("=" * 80)

!python scripts/main_modified.py data/imagenette2-320 \
    -a resnet50 \
    -b 128 \
    --epochs 1 \
    --max-iters 110 \
    --warmup-iters 10 \
    --log-file results/envC_T4/resnet50_b128.csv \
    --gpu 0

# ============================================================================
# Run MobileNetV2 Experiment
# ============================================================================
print("=" * 80)
print("EXPERIMENT 5: MobileNetV2, Batch 128")
print("=" * 80)

!python scripts/main_modified.py data/imagenette2-320 \
    -a mobilenet_v2 \
    -b 128 \
    --epochs 1 \
    --max-iters 110 \
    --warmup-iters 10 \
    --log-file results/envC_T4/mobilenetv2_b128.csv \
    --gpu 0

# ============================================================================
# Download Results
# ============================================================================
print("=" * 80)
print("DOWNLOADING RESULTS")
print("=" * 80)

# List all result files
!ls -lh results/envC_T4/

# Zip results for download
!zip -r results_envC_T4.zip results/envC_T4/

print("\n✓ Results saved to: results_envC_T4.zip")
print("Download this file from the Files panel on the left")

# ============================================================================
# Quick Analysis
# ============================================================================
import pandas as pd
import glob

print("=" * 80)
print("QUICK RESULTS SUMMARY")
print("=" * 80)

csv_files = glob.glob('results/envC_T4/*.csv')

for csv_file in sorted(csv_files):
    df = pd.read_csv(csv_file)
    # Skip warmup iterations
    df_measured = df[df['iteration'] > 10]
    
    if len(df_measured) > 0:
        avg_time = df_measured['batch_time'].mean()
        avg_throughput = df_measured['throughput_img_per_sec'].mean()
        
        print(f"\n{csv_file.split('/')[-1]}:")
        print(f"  Avg time per iteration: {avg_time:.4f}s")
        print(f"  Avg throughput: {avg_throughput:.2f} images/sec")

# ============================================================================
# Quick Analysis
# ============================================================================
import pandas as pd
import glob

print("=" * 80)
print("QUICK RESULTS SUMMARY")
print("=" * 80)

csv_files = glob.glob('results/envC_T4/*.csv')

for csv_file in sorted(csv_files):
    df = pd.read_csv(csv_file)
    
    if len(df) > 0:
        # Our CSV columns: env, device, model, batch_size, iter_idx, iter_time_ms, imgs_per_sec, timestamp
        avg_time_ms = df['iter_time_ms'].mean()
        avg_throughput = df['imgs_per_sec'].mean()
        num_iters = len(df)
        
        print(f"\n{csv_file.split('/')[-1]}:")
        print(f"  Model: {df['model'].iloc[0]}")
        print(f"  Batch size: {df['batch_size'].iloc[0]}")
        print(f"  Device: {df['device'].iloc[0]}")
        print(f"  Iterations logged: {num_iters}")
        print(f"  Avg time per iteration: {avg_time_ms:.2f} ms ({avg_time_ms/1000:.4f}s)")
        print(f"  Avg throughput: {avg_throughput:.2f} images/sec")
