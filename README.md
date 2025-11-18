# CNN Training Performance Analysis and Roofline Modeling

**Author:** Dhairya Mishra (dpm873@nyu.edu)  
**Project:** Cloud Computing Performance Analysis - ImageNet CNN Training

This project analyzes the performance of three CNN models (ResNet18, ResNet50, MobileNetV2) across three compute environments (RTX 4080 GPU, T4 GPU, CPU) using roofline modeling to identify performance bottlenecks.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dataset Setup](#dataset-setup)
5. [Running Experiments](#running-experiments)
6. [Analysis Scripts](#analysis-scripts)
7. [Project Structure](#project-structure)
8. [Results](#results)

---

## Project Overview

**Goal:** Quantitatively compare CNN training performance across different hardware environments and analyze behavior using roofline models.

**Environments:**
- Environment A: RTX 4080 Mobile GPU (12GB VRAM)
- Environment B: Intel i9-14900HX CPU (32GB RAM)
- Environment C: Google Colab T4 GPU (16GB VRAM)

**Models:**
- ResNet18 (baseline, 11.7M parameters)
- ResNet50 (deeper, 25.6M parameters)
- MobileNetV2 (lightweight, 3.5M parameters)

**Key Metrics:**
- Training throughput (images/second)
- Time per iteration (milliseconds)
- Achieved FLOPs/s
- Hardware efficiency (percentage of theoretical peak)
- Arithmetic intensity (FLOPs/Byte)

---

## Prerequisites

**Hardware Requirements:**
- GPU with CUDA support (for GPU experiments)
- At least 8GB RAM (16GB+ recommended)
- 10GB free disk space for dataset

**Software Requirements:**
- Python 3.8+
- CUDA 11.0+ (for GPU experiments)
- Git

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/dhairyamishra/cloud-training-and-analysis-nyu25.git
cd cloud-training-and-analysis-nyu25
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- torch (PyTorch)
- torchvision
- fvcore (FLOPs counting)
- matplotlib, seaborn (visualization)
- pandas, numpy (data analysis)

### Step 4: Verify Installation

```bash
python gather_system_info.py
```

This will create `system_specifications.json` and `system_specifications.txt` with your hardware specs.

---

## Dataset Setup

### Option 1: Automatic Download (Recommended)

```bash
python download_imagenette.py
```

This downloads ImageNette (10-class ImageNet subset, ~1.5GB) to `data/imagenette2-320/`.

### Option 2: Manual Download

1. Download from: https://github.com/fastai/imagenette
2. Extract to `data/imagenette2-320/`
3. Verify structure:
   ```
   data/imagenette2-320/
   ├── train/
   │   ├── n01440764/
   │   ├── n02102040/
   │   └── ...
   └── val/
       ├── n01440764/
       ├── n02102040/
       └── ...
   ```

---

## Running Experiments

### Environment A: RTX 4080 GPU (Local)

**1. ResNet18 (batch 128):**
```bash
python scripts/main_modified.py data/imagenette2-320 \
    -a resnet18 -b 128 --epochs 1 --max-iters 110 \
    --warmup-iters 10 --log-file results/envA_4080/resnet18_b128.csv --gpu 0
```

**2. ResNet50 (batch 32, 64, 128):**
```bash
# Batch 32
python scripts/main_modified.py data/imagenette2-320 \
    -a resnet50 -b 32 --epochs 1 --max-iters 110 \
    --warmup-iters 10 --log-file results/envA_4080/resnet50_b32.csv --gpu 0

# Batch 64
python scripts/main_modified.py data/imagenette2-320 \
    -a resnet50 -b 64 --epochs 1 --max-iters 110 \
    --warmup-iters 10 --log-file results/envA_4080/resnet50_b64.csv --gpu 0

# Batch 128
python scripts/main_modified.py data/imagenette2-320 \
    -a resnet50 -b 128 --epochs 1 --max-iters 110 \
    --warmup-iters 10 --log-file results/envA_4080/resnet50_b128.csv --gpu 0
```

**3. MobileNetV2 (batch 128):**
```bash
python scripts/main_modified.py data/imagenette2-320 \
    -a mobilenet_v2 -b 128 --epochs 1 --max-iters 110 \
    --warmup-iters 10 --log-file results/envA_4080/mobilenet_v2_b128.csv --gpu 0
```

### Environment B: CPU (Local)

```bash
python scripts/main_modified.py data/imagenette2-320 \
    -a resnet18 -b 64 --epochs 1 --max-iters 60 \
    --warmup-iters 10 --log-file results/envB_cpu/resnet18_b64.csv --cpu
```

**Note:** CPU training is very slow. Only ResNet18 with batch 64 is recommended.

### Environment C: Google Colab T4 GPU

**1. Open Google Colab:**
- Go to: https://colab.research.google.com/
- Create new notebook

**2. Enable GPU:**
- Runtime > Change runtime type
- Hardware accelerator: GPU (T4)
- Save

**3. Run Setup Script:**

Copy the contents of `colab_setup.py` into Colab cells and execute them in order. The script will:
- Verify T4 GPU is available
- Clone this repository
- Download ImageNette dataset
- Run all experiments
- Save results for download

**4. Download Results:**

After experiments complete, download the results:
```python
# In Colab
from google.colab import files
import shutil

shutil.make_archive('results_envC_t4', 'zip', 'results/envC_t4')
files.download('results_envC_t4.zip')
```

---

## Analysis Scripts

After running experiments, analyze the results using these scripts:

### 1. Model Complexity Analysis

Calculate FLOPs and parameters for each model:

```bash
python scripts/complexity_analysis.py
```

**Output:**
- `analysis/model_complexity.csv` - Per-model complexity
- `analysis/batch_complexity.csv` - Complexity per batch size
- `analysis/arithmetic_intensity.csv` - AI calculations

### 2. Aggregate Results

Combine all experiment CSV files into summary statistics:

```bash
python analysis/aggregate_results.py
```

**Output:**
- `analysis/output/summary_statistics.csv` - Mean, std, min, max for all experiments

### 3. Calculate Performance Metrics

Compute achieved FLOPs/s and efficiency:

```bash
python analysis/calculate_performance_metrics.py
```

**Output:**
- `analysis/output/achieved_performance.csv` - FLOPs/s and efficiency
- `analysis/output/roofline_analysis.csv` - Roofline classification

### 4. Create Visualizations

Generate performance comparison plots:

```bash
python analysis/create_visualizations.py
```

**Output (in `analysis/output/`):**
- `throughput_comparison.png` - Throughput across environments
- `batch_size_scaling.png` - Batch size effects
- `efficiency_comparison.png` - Hardware efficiency
- `model_comparison.png` - Model performance

### 5. Create Roofline Plots

Generate roofline model visualizations:

```bash
python analysis/create_roofline_plots.py
```

**Output (in `analysis/output/`):**
- `roofline_rtx4080.png` - RTX 4080 roofline
- `roofline_t4.png` - T4 GPU roofline
- `roofline_cpu.png` - CPU roofline
- `roofline_combined.png` - All environments

---

## Project Structure

```
cloud-training-and-analysis-nyu25/
│
├── README.md                          # This file
├── ANALYSIS_REPORT.md                 # Final analysis report
├── requirements.txt                   # Python dependencies
├── project-guidelines.md              # Project requirements
│
├── scripts/                           # Training scripts
│   ├── main_modified.py               # Modified PyTorch ImageNet example
│   ├── complexity_analysis.py         # FLOPs/parameter counting
│   └── README.md                      # Script documentation
│
├── analysis/                          # Analysis scripts
│   ├── aggregate_results.py           # Combine experiment results
│   ├── calculate_performance_metrics.py  # FLOPs/s, efficiency
│   ├── create_visualizations.py       # Performance plots
│   ├── create_roofline_plots.py       # Roofline visualizations
│   └── output/                        # Generated plots and CSVs
│
├── results/                           # Experiment results
│   ├── envA_4080/                     # RTX 4080 results
│   ├── envB_cpu/                      # CPU results
│   └── envC_t4/                       # T4 GPU results
│
├── data/                              # Dataset directory
│   └── imagenette2-320/               # ImageNette dataset
│
├── docs/                              # Documentation
│
├── download_imagenette.py             # Dataset download script
├── download_pytorch_example.py        # Get PyTorch example
├── gather_system_info.py              # Hardware specs collection
├── setup_project_structure.py         # Create directory structure
├── setup_kaggle.py                    # Kaggle API setup
└── colab_setup.py                     # Google Colab setup script
```

---

## Results

All results are documented in `ANALYSIS_REPORT.md`.

**Key Findings:**

1. **Environment Comparison:**
   - GPUs are 60-100x faster than CPU
   - RTX 4080 generally 1.7-3.2x faster than T4
   - T4 achieved better efficiency (21% vs 7%)

2. **Model Complexity:**
   - All models are compute-bound (AI > ridge points)
   - ResNet50 has highest AI (238 FLOPs/Byte)
   - Even MobileNetV2 is compute-bound in training

3. **Batch Size Effects:**
   - Throughput increases with batch size
   - Diminishing returns after batch 64
   - RTX 4080 shows severe degradation at batch 128 (thermal throttling)

4. **Roofline Analysis:**
   - All workloads in compute-bound region
   - Low efficiency (7-36%) indicates optimization opportunities
   - Framework overhead significant

**View Results:**
- Summary statistics: `analysis/output/summary_statistics.csv`
- Performance metrics: `analysis/output/achieved_performance.csv`
- Visualizations: `analysis/output/*.png`

---

## Troubleshooting

**Issue: CUDA out of memory**
- Solution: Reduce batch size or use gradient accumulation

**Issue: Dataset not found**
- Solution: Run `python download_imagenette.py` or verify `data/imagenette2-320/` exists

**Issue: Import errors**
- Solution: Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

**Issue: Slow CPU training**
- Solution: Reduce `--max-iters` to 60 or skip CPU experiments

**Issue: No GPU detected**
- Solution: Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`






