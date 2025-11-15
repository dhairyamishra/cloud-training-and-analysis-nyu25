# Project 1 – ImageNet Performance Modeling and Roofline Analysis

This README documents the **full plan** for Project 1: performance modeling and roofline analysis using ImageNet-style CNN training.

It describes:

- Environments and models to use
- Experimental design & hypotheses
- Complexity estimation strategy
- Measurement and logging
- Roofline modeling (including optional Nsight usage)
- Analysis questions to answer
- A step-by-step task checklist to ensure all project requirements are covered

---

## 1. Experiment Overview

**Goal:**  
Quantitatively compare the performance of several ImageNet CNN models across different compute environments (CPU-only, local laptop GPU, and cloud GPU) using the PyTorch ImageNet example, and analyze their behavior using roofline models.

**High-level idea:**

- Use a **subset of ImageNet-1k** as the workload.
- Train (short runs only) several CNNs (ResNet18, ResNet50, MobileNetV2).
- Run the same training loop in three environments:
  - **Env A (Bare-metal GPU):** local laptop RTX 4080 GPU.
  - **Env B (CPU-only):** same laptop, PyTorch restricted to CPU.
  - **Env C (Cloud GPU):** Google Colab with T4 GPU.
- Collect timing + throughput + profiling metrics.
- Build **roofline models** for each GPU environment and place each model/configuration on those plots.
- Analyze how environment and model choice affect performance and roofline position.

Rubric mapping:

- **Experiment Design – 10%**
- **Complexity Estimation – 20%**
- **Measurement – 15%**
- **Roofline Modeling – 20%**
- **Analysis – 35%**

---

## 2. Environments

### Environment A – Bare-metal GPU (Local Laptop)

- **CPU:** Intel i9-14900HX
- **GPU:** NVIDIA RTX 4080 Laptop GPU, 12 GB VRAM
- **OS / Software:** (to be filled in)
  - Windows / WSL / Linux + CUDA + cuDNN
  - PyTorch + torchvision versions
  - Python version

Used for:

- Primary GPU performance measurements.
- Nsight Compute profiling and roofline extraction.

### Environment B – CPU-only (Same Laptop)

- **CPU:** Intel i9-14900HX
- **Device:** PyTorch `device="cpu"`
- Same OS / Python / PyTorch as Env A.

Used for:

- Contrasting CPU vs GPU performance on identical hardware.
- Simpler CPU roofline (approximate) for discussion.

### Environment C – Cloud GPU (Google Colab T4)

- **Runtime:** Google Colab (T4 GPU)
- **GPU:** NVIDIA T4, 16 GB VRAM
- **Software:**
  - Colab’s preinstalled CUDA + PyTorch (record versions).
  - Notebook to run the same ImageNet script, pointing to ImageNet subset on Google Drive.

Used for:

- “Cloud vs Bare-metal” comparison.
- Additional roofline for a data-center-class GPU.

---

## 3. Neural Network Models

Use torchvision models supported by the PyTorch ImageNet example:

1. **ResNet18** – small / baseline model.
2. **ResNet50** – deeper, more compute-intensive model.
3. **MobileNetV2** – lightweight, parameter-efficient model.

These three give a range of FLOPs, parameter counts, and memory behaviors, which is ideal for roofline analysis.

---

## 4. Dataset Setup (ImageNet Subset)

The full ImageNet-1k dataset is too large for this project. We only need a **representative subset**:

1. Start from an existing ImageNet-1k copy or download the training data.
2. Create a subset, e.g.:
   - Choose **N classes** (e.g., 20 classes).
   - Within each selected class, keep only **M images** (e.g., 500 per class).
3. Maintain the standard `ImageFolder` layout:
   - `train/class_x/xxx.png`
   - `val/class_x/yyy.png`
4. Copy the same subset to:
   - Local laptop (for Envs A & B).
   - Google Drive (for Env C) – or upload directly from local.

Document exactly how the subset was created in the report (number of classes, images per class, total images).

---

## 5. Experiment Design (Rubric: Experiment Design – 10%)

### 5.1 Objectives

> **Objective:** Quantify and compare the performance of ResNet18, ResNet50, and MobileNetV2 on an ImageNet subset across CPU-only, local RTX 4080 GPU, and cloud T4 GPU environments. Use roofline models to analyze whether workloads are compute- or memory-bound and to explain performance differences.

### 5.2 Hypotheses

1. **H1 (Environment):**  
   For the same model and batch size, both GPUs (RTX 4080 and T4) will significantly outperform the CPU. The RTX 4080 will achieve higher FLOPs/s than T4 due to higher compute peak, though differences in memory bandwidth may affect how close each GPU gets to its roofline.

2. **H2 (Model Complexity):**  
   Heavier models (ResNet50) will have higher arithmetic intensity and tend to be more compute-bound; lighter models (MobileNetV2) will be more memory-bound and will underutilize peak compute on high-end GPUs.

3. **H3 (Batch Size):**  
   Increasing batch size (up to memory limits) will improve arithmetic intensity and bring performance closer to the compute roofline, but with diminishing returns beyond a certain batch size.

### 5.3 Factor Design

- **Independent variables:**
  - Environment ∈ {Env A – 4080 GPU, Env B – CPU, Env C – T4 GPU}
  - Model ∈ {ResNet18, ResNet50, MobileNetV2}
  - Batch size:
    - Main comparison: batch size 128 on GPUs, 64 on CPU.
    - Batch sweep (for roofline / AI study): ResNet50 on 4080 with batch ∈ {32, 64, 128, 256}.

- **Dependent variables (metrics):**
  - Time per iteration (ms).
  - Throughput (images/second).
  - Achieved FLOPs/s (from FLOP estimates + timing).
  - Arithmetic intensity (FLOPs/byte).
  - For GPUs: roofline-related metrics (via Nsight metrics on Env A; approximate for Env C).

---

## 6. Complexity Estimation (Rubric: 20%)

For each model:

1. **Static quantities:**
   - Total number of parameters.
   - FLOPs per forward pass for a single 224×224 image.
   - Use either:
     - A FLOPs counting tool (e.g., `fvcore.nn.FlopCountAnalysis`), or
     - Published FLOP counts for standard architectures.

2. **Training FLOPs per image:**
   - Approximate training FLOPs ≈ `3 × (inference FLOPs)` (forward + backward + some overhead).

3. **Training FLOPs per iteration:**
   - `FLOPs_per_iter = FLOPs_per_image_train × batch_size`.

4. **Complexity table:**
   - For each `(model, batch_size)` pair, create a table containing:
     - Parameters
     - FLOPs/image (inference)
     - FLOPs/image (training approx)
     - FLOPs/iteration

This table feeds directly into roofline performance calculations.

---

## 7. Measurement Methodology (Rubric: 15%)

### 7.1 Training Script

Use the PyTorch ImageNet example (`main.py`) and minimally extend it:

- Add arguments:
  - `--max-iters N`: maximum training iterations before exit.
  - `--warmup-iters K`: number of warm-up iterations excluded from measurement.
  - `--log-file path`: CSV output file.
- Inside the training loop:
  - Skip logging for the first `warmup_iters`.
  - For each measured iteration:
    - Record start and end time.
    - Compute per-iteration time and throughput.
    - Append row to CSV with fields:
      - `env, device, model, batch_size, iter_idx, iter_time_ms, imgs_per_sec`.

### 7.2 Run Protocol

For each configuration:

1. Run **K warm-up iterations** (e.g., K = 10).
2. Then run **N measured iterations** (e.g., N = 100 on GPUs, 50–100 on CPU).
3. Use a fixed random seed for reproducibility (optional).
4. Store logs in `results/` directory, with filenames encoding env/model/batch.

Example GPU run (Env A):

```bash
python main.py \
  -a resnet50 \
  --batch-size 128 \
  --epochs 1 \
  --device cuda \
  --max-iters 110 \
  --warmup-iters 10 \
  --data /path/to/imagenet_subset \
  --log-file results/envA_4080_resnet50_bs128.csv

```

---

## 8. Task Checklist - Project Implementation

This section provides a complete, actionable task list to implement the project. Mark each item as complete (`[x]`) as you finish it.

### **Phase 1: Project Setup & Environment Configuration**

- [x] **Task 1.1:** Document system specifications
  - [x] Record OS version, CUDA version, cuDNN version
  - [x] Record PyTorch and torchvision versions
  - [x] Record Python version
  - [x] Document RTX 4080 specifications (CUDA cores, memory bandwidth, peak FLOPs)
  - [x] Document T4 GPU specifications (for comparison)

- [x] **Task 1.2:** Set up project directory structure
  - [x] Create `scripts/` directory for training scripts
  - [x] Create `data/` directory for dataset
  - [x] Create `results/` directory for experiment logs
  - [x] Create `analysis/` directory for analysis notebooks/scripts
  - [x] Create `roofline/` directory for roofline data and plots
  - [x] Create `docs/` directory for report and documentation

- [x] **Task 1.3:** Set up Python environment
  - [x] Create virtual environment or conda environment
  - [x] Install PyTorch with CUDA support
  - [x] Install torchvision
  - [x] Install required packages: numpy, pandas, matplotlib, seaborn
  - [x] Install FLOPs counting tool (fvcore or thop)
  - [x] Create `requirements.txt` file

### **Phase 2: Dataset Preparation**

- [x] **Task 2.1:** Obtain ImageNet-1k dataset
  - [x] Download or access existing ImageNet-1k dataset
  - [x] Verify dataset integrity

- [x] **Task 2.2:** Create ImageNet subset *(SKIPPED - ImageNette is already a perfect subset)*
  - [x] ~~Write script to create subset~~ (Not needed - using ImageNette)
  - [x] ~~Select N classes~~ (ImageNette has 10 classes)
  - [x] ~~Select M images per class~~ (ImageNette: ~963 train, ~387 val per class)
  - [x] ~~Maintain ImageFolder structure~~ (ImageNette already in correct format)
  - [x] ~~Run subset creation script~~ (Not needed)
  - [x] ~~Verify subset structure~~ (Already verified in Task 2.1)
  - [x] **Note:** ImageNette is already a perfect subset, so this task was skipped.

- [x] **Task 2.3:** Prepare dataset for different environments
  - [x] Copy subset to local directory for Env A & B
  - [x] Upload subset to Google Drive for Env C (Colab)
  - [x] Document dataset paths for each environment

### **Phase 3: Complexity Estimation (Rubric: 20%)**

- [x] **Task 3.1:** Set up FLOPs counting
  - [x] Create script to count FLOPs for each model (`complexity_analysis.py`)
  - [x] Test FLOPs counting with dummy input (224×224×3)

- [x] **Task 3.2:** Calculate model complexities
  - [x] Count parameters for ResNet18
  - [x] Count parameters for ResNet50
  - [x] Count parameters for MobileNetV2
  - [x] Calculate inference FLOPs for ResNet18
  - [x] Calculate inference FLOPs for ResNet50
  - [x] Calculate inference FLOPs for MobileNetV2

- [x] **Task 3.3:** Estimate training complexities
  - [x] Calculate training FLOPs per image (≈ 3× inference FLOPs)
  - [x] Calculate FLOPs per iteration for different batch sizes (32, 64, 128, 256)
  - [x] Create complexity table with all models and batch sizes

- [x] **Task 3.4:** Calculate arithmetic intensity estimates
  - [x] Estimate memory traffic per iteration for each model
  - [x] Calculate arithmetic intensity (FLOPs/byte) for each configuration
  - [x] Document assumptions made in calculations

### **Phase 4: Training Script Modification (Rubric: 15% - Measurement)**

- [x] **Task 4.1:** Obtain PyTorch ImageNet example
  - [x] Download/clone PyTorch examples repository
  - [x] Locate `imagenet/main.py` script
  - [x] Copy to project `scripts/` directory

- [x] **Task 4.2:** Extend training script with measurement capabilities
  - [x] Add `--max-iters` argument to limit training iterations
  - [x] Add `--warmup-iters` argument for warm-up period
  - [x] Add `--log-file` argument for CSV output path
  - [x] Add timing instrumentation in training loop
  - [x] Implement per-iteration timing (start/end timestamps)
  - [x] Calculate throughput (images/second)
  - [x] Implement CSV logging with fields: env, device, model, batch_size, iter_idx, iter_time_ms, imgs_per_sec

- [x] **Task 4.3:** Test modified training script
  - [x] Test on small dataset with CPU
  - [x] Test on small dataset with GPU
  - [x] Verify CSV output format
  - [x] Verify timing accuracy

### **Phase 5: Experiment Execution - Environment A (RTX 4080 GPU)**

- [x] **Task 5.1:** Run ResNet18 experiments on RTX 4080
  - [x] Run ResNet18, batch size 128, 110 iters (10 warmup + 100 measured)
  - [x] Verify log file created successfully
  - [x] Check for any errors or warnings

- [x] **Task 5.2:** Run ResNet50 experiments on RTX 4080
  - [x] Run ResNet50, batch size 128, 110 iters
  - [x] Run ResNet50, batch size 32 (for batch sweep)
  - [x] Run ResNet50, batch size 64 (for batch sweep)
  - [x] Run ResNet50, batch size 256 (SKIPPED - Out of Memory on 12GB VRAM)

- [x] **Task 5.3:** Run MobileNetV2 experiments on RTX 4080
  - [x] Run MobileNetV2, batch size 128, 110 iters
  - [x] Verify log file created successfully

- [x] **Task 5.4:** Collect Nsight Compute profiling data (for roofline)
  - [x] SKIPPED - Nsight Compute not installed
  - [x] Will use theoretical specs + measured performance instead

### **Phase 6: Experiment Execution - Environment B (CPU-only)**

- [x] **Task 6.1:** Run ResNet18 on CPU
  - [x] SKIPPED - Too slow (~40s per iteration on local CPU)
  - [x] Used Colab CPU run as reference (1 partial experiment completed)

- [x] **Task 6.2:** Run ResNet50 on CPU
  - [x] SKIPPED - Too slow for practical completion

- [x] **Task 6.3:** Run MobileNetV2 on CPU
  - [x] SKIPPED - Too slow for practical completion

**Note:** CPU experiments deemed impractical due to extreme slowness. One partial Colab CPU run provides sufficient comparison data.

### **Phase 7: Experiment Execution - Environment C (Google Colab T4)**

- [x] **Task 7.1:** Set up Google Colab environment
  - [x] Create Colab setup script (colab_setup.py)
  - [x] Clone repository in Colab
  - [x] Download ImageNette dataset in Colab
  - [x] Verify T4 GPU enabled

- [x] **Task 7.2:** Run ResNet18 on T4 GPU
  - [x] Run ResNet18, batch size 128, 110 iters (10 warmup + 100 measured)
  - [x] Results: 238ms/iter, 650 img/s
  - [x] Download log file from Colab

- [x] **Task 7.3:** Run ResNet50 on T4 GPU (batch sweep)
  - [x] Run ResNet50, batch size 32: 314ms/iter, 111 img/s
  - [x] Run ResNet50, batch size 64: 605ms/iter, 117 img/s
  - [x] Run ResNet50, batch size 128: 1208ms/iter, 120 img/s
  - [x] Download log files from Colab

- [x] **Task 7.4:** Run MobileNetV2 on T4 GPU
  - [x] Run MobileNetV2, batch size 128, 110 iters
  - [x] Results: 446ms/iter, 308 img/s
  - [x] Download log file from Colab

- [x] **Task 7.5:** Collect T4 profiling data
  - [x] SKIPPED - Nsight profiling not available on Colab
  - [x] Document T4 specifications (peak FLOPs, memory bandwidth)
  - [x] Will use theoretical specs for roofline analysis

### **Phase 8: Data Processing & Analysis**

- [ ] **Task 8.1:** Aggregate experimental results
  - [ ] Combine all CSV log files into single dataframe
  - [ ] Calculate summary statistics (mean, std, min, max) for each configuration
  - [ ] Create performance summary table

- [ ] **Task 8.2:** Calculate achieved performance metrics
  - [ ] Calculate achieved FLOPs/s for each configuration (using complexity estimates + timing)
  - [ ] Calculate achieved memory bandwidth (where applicable)
  - [ ] Calculate arithmetic intensity for each configuration

- [ ] **Task 8.3:** Create performance comparison visualizations
  - [ ] Bar chart: throughput (imgs/sec) across all configurations
  - [ ] Bar chart: time per iteration across all configurations
  - [ ] Line plot: batch size vs throughput (for ResNet50 batch sweep)
  - [ ] Grouped comparison: same model across different environments

### **Phase 9: Roofline Modeling (Rubric: 20%)**

- [ ] **Task 9.1:** Gather GPU specifications
  - [ ] Document RTX 4080 peak FP32 FLOPs/s
  - [ ] Document RTX 4080 memory bandwidth (GB/s)
  - [ ] Document T4 peak FP32 FLOPs/s
  - [ ] Document T4 memory bandwidth (GB/s)

- [ ] **Task 9.2:** Create roofline model for RTX 4080
  - [ ] Write script to generate roofline plot (`roofline_plot.py`)
  - [ ] Plot compute roofline (peak FLOPs/s)
  - [ ] Plot memory roofline (bandwidth × arithmetic intensity)
  - [ ] Mark ridge point (where compute and memory bounds intersect)

- [ ] **Task 9.3:** Plot experimental points on RTX 4080 roofline
  - [ ] Plot ResNet18 (batch 128) on roofline
  - [ ] Plot ResNet50 (batch 128) on roofline
  - [ ] Plot ResNet50 (batch 32, 64, 256) on roofline
  - [ ] Plot MobileNetV2 (batch 128) on roofline
  - [ ] Add labels and legend

- [ ] **Task 9.4:** Create roofline model for T4 GPU
  - [ ] Plot T4 compute and memory rooflines
  - [ ] Plot all T4 experimental points
  - [ ] Add labels and legend

- [ ] **Task 9.5:** Create CPU roofline (optional/approximate)
  - [ ] Document CPU peak FLOPs/s (estimate)
  - [ ] Document CPU memory bandwidth
  - [ ] Create simplified CPU roofline plot
  - [ ] Plot CPU experimental points

### **Phase 10: Analysis & Report Writing (Rubric: 35%)**

- [ ] **Task 10.1:** Analyze Environment comparison (H1)
  - [ ] Compare GPU vs CPU performance quantitatively
  - [ ] Compare RTX 4080 vs T4 performance
  - [ ] Explain differences using hardware specifications
  - [ ] Discuss roofline positioning differences

- [ ] **Task 10.2:** Analyze Model complexity effects (H2)
  - [ ] Compare ResNet18 vs ResNet50 vs MobileNetV2
  - [ ] Analyze which models are compute-bound vs memory-bound
  - [ ] Explain using arithmetic intensity and roofline position
  - [ ] Discuss efficiency (% of peak achieved)

- [ ] **Task 10.3:** Analyze Batch size effects (H3)
  - [ ] Analyze ResNet50 batch sweep results (32, 64, 128, 256)
  - [ ] Plot arithmetic intensity vs batch size
  - [ ] Plot performance vs batch size
  - [ ] Identify point of diminishing returns
  - [ ] Explain using roofline model

- [ ] **Task 10.4:** Answer key analysis questions
  - [ ] Which configurations are compute-bound vs memory-bound?
  - [ ] How close does each configuration get to theoretical peak?
  - [ ] What are the bottlenecks for each configuration?
  - [ ] How does batch size affect roofline position?
  - [ ] What optimizations could improve performance?

- [ ] **Task 10.5:** Write experiment design section (10%)
  - [ ] Document objectives clearly
  - [ ] State hypotheses (H1, H2, H3)
  - [ ] Describe factor design (independent/dependent variables)
  - [ ] Justify choices made

- [ ] **Task 10.6:** Write complexity estimation section (20%)
  - [ ] Present complexity table with all models
  - [ ] Show FLOPs calculations
  - [ ] Show arithmetic intensity calculations
  - [ ] Document methodology and assumptions

- [ ] **Task 10.7:** Write measurement methodology section (15%)
  - [ ] Describe training script modifications
  - [ ] Explain measurement protocol (warmup, iterations)
  - [ ] Show example commands used
  - [ ] Discuss measurement accuracy and reproducibility

- [ ] **Task 10.8:** Write roofline modeling section (20%)
  - [ ] Present roofline plots for each GPU
  - [ ] Explain roofline model construction
  - [ ] Interpret experimental points on rooflines
  - [ ] Discuss compute-bound vs memory-bound regions

- [ ] **Task 10.9:** Write analysis section (35%)
  - [ ] Present all performance comparison results
  - [ ] Analyze hypothesis H1 (environment effects)
  - [ ] Analyze hypothesis H2 (model complexity effects)
  - [ ] Analyze hypothesis H3 (batch size effects)
  - [ ] Provide insights and recommendations
  - [ ] Discuss limitations and future work

- [ ] **Task 10.10:** Create final report document
  - [ ] Write abstract/executive summary
  - [ ] Write introduction
  - [ ] Compile all sections
  - [ ] Add all figures and tables
  - [ ] Write conclusion
  - [ ] Add references
  - [ ] Proofread and format

### **Phase 11: Final Deliverables**

- [ ] **Task 11.1:** Organize code repository
  - [ ] Clean up all scripts
  - [ ] Add comments and documentation
  - [ ] Create README for repository
  - [ ] Ensure reproducibility

- [ ] **Task 11.2:** Package results
  - [ ] Organize all log files in `results/`
  - [ ] Save all plots in `roofline/` and `analysis/`
  - [ ] Create summary spreadsheet with all metrics

- [ ] **Task 11.3:** Final review
  - [ ] Verify all rubric items are addressed
  - [ ] Check that all hypotheses are tested
  - [ ] Ensure all required plots are included
  - [ ] Verify calculations are correct

- [ ] **Task 11.4:** Submit project
  - [ ] Prepare submission package
  - [ ] Submit report
  - [ ] Submit code repository
  - [ ] Submit any additional required materials

---

## 9. Progress Tracking

**Total Tasks:** 100+  
**Completed:** 35 (Phases 1-7 complete)
**In Progress:** Phase 8 (Data Analysis)
**Remaining:** ~65 (Phases 8-11)

**Current Phase:** Phase 8 - Data Processing & Analysis

**Experiment Status:**
- Environment A (RTX 4080): 6 experiments complete
- Environment B (CPU): 1 partial experiment (skipped - too slow)
- Environment C (T4 GPU): 5 experiments complete
- **Total:** 12 CSV files with timing/throughput data

**Last Updated:** 2025-11-14

---

## 10. Notes and Observations

Use this section to document any issues, insights, or deviations from the plan as you work through the project.

### **Key Deviations from Original Plan:**

1. **Dataset Choice:** Used ImageNette (10-class subset) instead of creating custom ImageNet subset
   - ImageNette is already optimized and well-structured
   - 342 MB download vs potential 10+ GB for custom subset
   - ~9,630 training images, ~3,870 validation images

2. **CPU Experiments Skipped:** Environment B (CPU-only) experiments were largely skipped
   - Reason: Extremely slow performance (~40s per iteration vs ~0.2s on GPU)
   - Would take 2-3 hours per experiment
   - One partial ResNet18 run completed for reference
   - Colab CPU run provides additional CPU comparison data

3. **Nsight Compute Profiling Skipped:** Task 5.4 skipped
   - Nsight Compute not installed on system
   - Will use theoretical GPU specs + measured performance for roofline analysis
   - Sufficient for project requirements

4. **ResNet50 Batch 256 on RTX 4080:** Out of Memory
   - RTX 4080 has 12GB VRAM (laptop version)
   - Batch 256 exceeded memory capacity
   - Successfully ran batch 32, 64, 128 for batch sweep analysis

### **Key Findings (Preliminary):**

**RTX 4080 vs T4 Performance:**
- RTX 4080 consistently faster: 1.4x - 2.7x speedup
- Largest gap on ResNet50 (2.67x) - compute-intensive model benefits from RTX 4080's higher compute
- Smallest gap on ResNet18 (1.36x) - lighter model, less compute-bound

**Batch Size Effects (ResNet50 on T4):**
- Batch 32: 113 img/s
- Batch 64: 117 img/s  
- Batch 128: 120 img/s
- Diminishing returns observed - throughput plateaus as batch size increases
- Suggests memory bandwidth becoming bottleneck at larger batches

**Model Comparison (T4 GPU, Batch 128):**
- ResNet18: 649 img/s (fastest, lightest)
- MobileNetV2: 308 img/s (efficient but slower than expected)
- ResNet50: 120 img/s (slowest, most compute-intensive)

### **Technical Issues Resolved:**

1. **CPU Mode Bug in main_modified.py:**
   - Issue: `--no-accel` flag not properly respected
   - Model and criterion still placed on CUDA even with flag set
   - Fixed: Added proper device placement logic for CPU mode
   - Validation function also updated to handle CPU tensors

2. **Colab Setup Script:**
   - Added GPU verification cell to prevent accidental CPU runs
   - Script now stops execution if GPU not enabled
   - Clear error messages guide user to enable GPU

3. **Git Ignore Configuration:**
   - Initially blocked all CSV files in results/
   - Updated to allow CSV files (small, important for analysis)
   - Still blocks large files (datasets, model checkpoints, logs)

---

## 11. Quick Reference

### Key Commands

**Environment A (RTX 4080 GPU):**
```bash
python scripts/main.py -a resnet50 --batch-size 128 --epochs 1 --device cuda \
  --max-iters 110 --warmup-iters 10 --data data/imagenet_subset \
  --log-file results/envA_4080_resnet50_bs128.csv
```

**Environment B (CPU):**
```bash
python scripts/main.py -a resnet50 --batch-size 64 --epochs 1 --device cpu \
  --max-iters 60 --warmup-iters 10 --data data/imagenet_subset \
  --log-file results/envB_cpu_resnet50_bs64.csv
```

### Important Paths
- Dataset: `data/imagenet_subset/`
- Scripts: `scripts/`
- Results: `results/`
- Analysis: `analysis/`
- Roofline: `roofline/`

### Model Names
- `resnet18`
- `resnet50`
- `mobilenet_v2`
