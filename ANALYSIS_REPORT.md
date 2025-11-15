# Performance Modeling and Roofline Analysis of CNN Training
## ImageNet-Style Training Across Multiple Compute Environments

**Author:** Dhairya Mishra - dpm873@nyu.edu 
**Date:** November 14, 2025  
**Course:** Cloud Computing Performance Analysis

---

## Executive Summary

In this project, I trained three popular CNN models (ResNet18, ResNet50, and MobileNetV2) on a subset of ImageNet data across three different computing environments: my laptop's RTX 4080 GPU, Google Colab's T4 GPU, and CPU-only mode. The goal was to understand how different hardware affects training performance and to use roofline modeling to identify bottlenecks.

**What I Found:**

The results were quite surprising! The cloud-based T4 GPU actually achieved better efficiency (21% on average) than my more powerful RTX 4080 (only 7% efficiency). While the RTX 4080 was generally 1.4-2.7× faster, it severely underutilized its theoretical capabilities. All my experiments turned out to be compute-bound rather than memory-bound, and I discovered that increasing batch size gives diminishing returns. Most interestingly, ResNet50 with batch size 128 on the RTX 4080 showed a dramatic performance collapse (only 0.61% efficiency) - likely due to thermal throttling on my laptop.

---

## 1. Experiment Design (10%)

### 1.1 What I Set Out to Do

**Main Goal:** I wanted to understand how different hardware environments affect CNN training performance. Specifically, I was curious about whether expensive hardware always means better performance, and where the bottlenecks really are - is it compute power or memory bandwidth?

**What I Hoped to Learn:**
1. How much faster are GPUs compared to CPUs in real-world training?
2. Can I measure the actual FLOPs/s achieved versus the theoretical peak?
3. Where do my workloads sit on the roofline model - are they compute-bound or memory-bound?
4. What's the sweet spot for batch size?

### 1.2 My Hypotheses

Going into this, I had three main predictions:

**H1 - Hardware Matters:** I expected GPUs to crush the CPU in performance, and my RTX 4080 (with its impressive 48.7 TFLOPs/s peak) to outperform the T4 (8.1 TFLOPs/s). This seemed obvious given the specs.

**H2 - Model Complexity Matters:** I thought heavier models like ResNet50 would be compute-bound (lots of math operations), while lighter models like MobileNetV2 would be memory-bound (waiting for data). This is what textbooks suggest.

**H3 - Bigger Batches Help... Until They Don't:** I predicted that increasing batch size would improve performance by doing more work per memory access, but eventually you'd hit a wall due to memory limitations.

### 1.4 My Test Environments

Here's what I had to work with:

| Environment | What It Is | Peak Power | Memory Speed | Memory Size |
|-------------|-----------|------------|--------------|-------------|
| **My Laptop (Env A)** | RTX 4080 Mobile GPU | 48.7 TFLOPs/s | 384 GB/s | 12 GB |
| **CPU Mode (Env B)** | Intel i9-14900HX | ~1.5 TFLOPs/s | 89.6 GB/s | 32 GB (RAM) |
| **Google Colab (Env C)** | T4 GPU (cloud) | 8.1 TFLOPs/s | 300 GB/s | 16 GB |

The RTX 4080 looks dominant on paper - 6× more compute power than the T4! But results may vary
---

## 2. Complexity Estimation (20%)

### 2.1 Understanding the Models

First, I needed to understand what I was dealing with. Here's how complex each model is:

| Model | Parameters | Operations per Image | Training Operations |
|-------|-----------|---------------------|---------------------|
| **ResNet18** | 11.7M | 1.82 billion FLOPs | 5.46 billion FLOPs |
| **ResNet50** | 25.6M | 4.09 billion FLOPs | 12.27 billion FLOPs |
| **MobileNetV2** | 3.5M | 0.30 billion FLOPs | 0.90 billion FLOPs |

**Quick note:** Training takes about 3× more operations than inference because you need to do the forward pass, backward pass, and update the weights. I calculated these using PyTorch's built-in FLOPs counter.

---

### 2.3 Arithmetic Intensity - The Key Metric

This is where things get interesting. Arithmetic Intensity (AI) tells us how many operations we do per byte of data moved. High AI means compute-bound (good for GPUs), low AI means memory-bound (waiting for data).

**My ResNet50 Results:**

| Batch Size | AI (FLOPs/Byte) | Insight |
|------------|-----------------|---------|
| 32 | 227 | 8× above T4 ridge point (27) |
| 64 | 234 | AI increases with batch size |
| 128 | 238 | Still compute-bound, not memory-bound |

**What this means:** All my workloads had AI way above the "ridge points" (RTX 4080: 127, T4: 27). When AI is this high, the GPU should be limited by compute power, not memory bandwidth. This was my first hint that H2 might be wrong - even the "lightweight" models would be compute-bound!

---

## 3. Measurement Methodology (15%)

### 3.1 How I Modified the Code

I started with PyTorch's standard ImageNet training example and added my own instrumentation:

**What I Added:**
1. **Iteration limits** - No need to train forever, just enough to get good measurements
2. **Warmup period** - First 10 iterations excluded (GPU needs to "warm up")
3. **Detailed logging** - Every iteration logged to CSV with timing and throughput

The code was pretty straightforward - just wrap the training loop with timing calls and write everything to a CSV file.

### 3.2 My Testing Process

For each experiment, I followed the same protocol:
1. Run 10 warmup iterations (thrown away)
2. Run 100 measured iterations (50 for CPU - it was painfully slow)
3. Log everything: time per iteration, images per second, timestamps
4. Calculate statistics: mean, standard deviation, min, max

**Example of what I ran:**
```bash
python scripts/main_modified.py data/imagenette2-320 \
    -a resnet50 -b 128 --epochs 1 --max-iters 110 \
    --warmup-iters 10 --log-file results/envA_4080/resnet50_b128.csv --gpu 0
```

---

## 4. Roofline Modeling (20%)

### 4.1 Understanding the Roofline Model

The roofline model is a brilliant way to visualize performance. Imagine a graph where:
- X-axis: Arithmetic Intensity (how compute-heavy your workload is)
- Y-axis: Performance (TFLOPs/s achieved)

The "roofline" itself has two parts:
1. **A diagonal line** - memory-bound region (limited by bandwidth)
2. **A flat line** - compute-bound region (limited by peak FLOPs)

Where these meet is the "ridge point" - the transition between memory-bound and compute-bound.

### 4.2 RTX 4080 Results - The Disappointing Reality

My laptop's RTX 4080 should be a beast with 48.7 TFLOPs/s peak. Here's what I actually got:

| Model | Batch | AI | Achieved | Efficiency | My Reaction |
|-------|-------|----|-----------|-----------| ------------|
| ResNet18 | 128 | 230 | 5.5 TFLOPs/s | 11.4% | Disappointing |
| ResNet50 | 32 | 227 | 4.1 TFLOPs/s | 8.5% | Worse |
| ResNet50 | 64 | 234 | 4.2 TFLOPs/s | 8.6% | Still bad |
| ResNet50 | 128 | 238 | 0.3 TFLOPs/s | 0.6% | **What?!** |

That last one - ResNet50 with batch 128 - was shocking. Only 0.6% efficiency! Something was seriously wrong. More on this mystery later.

### 4.3 T4 GPU Results - The Underdog Wins

The T4 has much lower peak performance (8.1 TFLOPs/s), but look at these efficiency numbers:

| Model | Batch | AI | Achieved | Efficiency | My Reaction |
|-------|-------|----|-----------|-----------| ------------|
| ResNet18 | 128 | 230 | 2.9 TFLOPs/s | 36.2% | Impressive! |
| ResNet50 | 32 | 227 | 1.3 TFLOPs/s | 15.5% | Solid |
| ResNet50 | 64 | 234 | 1.3 TFLOPs/s | 16.1% | Consistent |
| ResNet50 | 128 | 238 | 1.3 TFLOPs/s | 16.1% | Stable |

The T4 achieved 2-3× better efficiency than my RTX 4080! This was completely unexpected and became one of the most interesting findings of this project.

---

## 5. Analysis (35%)

### 5.1 Testing H1: Does Expensive Hardware Always Win?

**What I Expected:** My RTX 4080 should dominate everything.

**What Actually Happened:** It's complicated.

#### GPU vs CPU - No Surprises Here

The GPUs absolutely destroyed the CPU:
- **CPU:** 10 images/second (ResNet18, batch 64)
- **RTX 4080:** 1,107 images/second (ResNet18, batch 128) - **107× faster!**
- **T4 GPU:** 650 images/second (ResNet18, batch 128) - **63× faster**

So yes, GPUs are essential for deep learning. No surprises there.

#### RTX 4080 vs T4 - The Plot Twist

Here's where things got interesting:

| Model | Batch | RTX 4080 | T4 | Speedup | Winner |
|-------|-------|----------|----|---------| -------|
| ResNet18 | 128 | 1,107 img/s | 650 img/s | 1.7× | RTX 4080 |
| ResNet50 | 32 | 359 img/s | 111 img/s | 3.2× | RTX 4080 |
| ResNet50 | 64 | 366 img/s | 117 img/s | 3.1× | RTX 4080 |
| ResNet50 | 128 | **31 img/s** | 120 img/s | 0.26× | **T4 wins?!** |
| MobileNetV2 | 128 | 752 img/s | 308 img/s | 2.4× | RTX 4080 |

Wait, what? The RTX 4080 lost to the T4 on ResNet50 batch 128? And not just lost - it was 4× slower! This became my biggest mystery to solve.

**The Efficiency Paradox:**
- RTX 4080 average efficiency: 7.3%
- T4 average efficiency: 21.0%

The T4 was 2.9× more efficient despite having 6× less peak compute power. Why?

**My Theories:**
1. **Thermal throttling** - My laptop GPU was probably overheating
2. **Memory pressure** - 12GB VRAM was getting tight with batch 128
3. **Framework overhead** - Maybe PyTorch/CUDA doesn't optimize well for this GPU
4. **Cloud advantage** - The T4 in Colab has better cooling and power delivery

**Verdict:** H1 is partially confirmed - RTX 4080 is generally faster, but with a huge caveat about the batch 128 anomaly.

---

### 5.2 Testing H2: Are Lighter Models Memory-Bound?

**What I Expected:** ResNet50 (heavy) would be compute-bound, MobileNetV2 (light) would be memory-bound.

**What Actually Happened:** Everything was compute-bound!

| Model | Parameters | AI | Ridge Point | Bound Type |
|-------|-----------|----|-----------| ------------|
| ResNet18 | 11.7M | 230 | 27 | Compute-bound |
| ResNet50 | 25.6M | 238 | 27 | Compute-bound |
| MobileNetV2 | 3.5M | ~220 | 27 | Compute-bound |

All my arithmetic intensity values were way above the ridge points. Even MobileNetV2, which I thought would be memory-bound, had AI > 220.

**Why Was I Wrong?**

After thinking about it, this makes sense:
1. **Training is different from inference** - Training has more compute per memory access
2. **Batching helps** - Processing 128 images at once amortizes memory costs
3. **Modern CNNs are optimized** - They're designed to be compute-efficient
4. **Ridge points are low** - Even the T4's ridge point is only 27 FLOPs/Byte

**Verdict:** H2 is not confirmed - All models are compute-bound, not memory-bound as I predicted.

### 5.3 Testing H3: Does Bigger Batch Size Always Help?

**What I Expected:** Batch size should improve performance until you hit memory limits.

**What Actually Happened:** Diminishing returns kick in fast!

**ResNet50 on T4 GPU:**

| Batch Size | AI | Throughput | Improvement |
|------------|----|-----------| ------------|
| 32 | 227 | 111 img/s | baseline |
| 64 | 234 | 117 img/s | +5.7% |
| 128 | 238 | 120 img/s | +2.1% |

Going from batch 32 to 64 gave me a nice 5.7% boost. But doubling again to 128 only added 2.1%. And on the RTX 4080, batch 128 actually made things worse!

**Why Diminishing Returns?**

1. **Parallelism saturates** - At some point, the GPU is fully utilized
2. **Memory bandwidth matters** - Even with high AI, you still need to move data
3. **Thermal limits** - Bigger batches mean more heat
4. **Synchronization overhead** - Larger batches need more coordination

**Verdict:** H3 is confirmed - Batch size helps, but with clear diminishing returns.

---

### 5.4 The RTX 4080 Mystery - What Went Wrong?

This deserves its own section because it was so dramatic.

**The Problem:**
- ResNet50 batch 32: 359 img/s (8.5% efficiency) - Good
- ResNet50 batch 64: 366 img/s (8.6% efficiency) - Good
- ResNet50 batch 128: **31 img/s (0.6% efficiency)** - Bad!

That's a 12× slowdown! The iteration times were also all over the place (standard deviation of 1,731ms, ranging from 914ms to 8,561ms).

**My Investigation:**

I think it's **thermal throttling**. Here's why:
1. **Laptop GPU** - Limited cooling compared to desktop or datacenter GPUs
2. **12GB VRAM limit** - Batch 128 pushes close to the limit
3. **High variance** - Suggests the GPU is throttling up and down
4. **Sustained load** - 100 iterations is enough time for heat to build up

The T4 in Google Colab doesn't have this problem because datacenter GPUs have much better cooling and power delivery.

**Lesson Learned:** Raw specs don't tell the whole story. A well-cooled 8.1 TFLOPs/s GPU can outperform a thermally-constrained 48.7 TFLOPs/s GPU in sustained workloads.

---

## 6. Conclusions

### 6.1 What I Learned

This project taught me that performance analysis is full of surprises:

**1. Environment Effects (H1):**
- GPUs are 60-100× faster than CPUs (no surprise)
- RTX 4080 is generally 1.7-3.2× faster than T4 (expected)
- But thermal throttling can completely reverse this (unexpected!)

**2. Model Complexity (H2):**
- All models ended up compute-bound (I was wrong)
- Heavier models do have slightly higher arithmetic intensity (partially right)
- The lesson: Training workloads are different from inference

**3. Batch Size Effects (H3):**
- Bigger batches improve arithmetic intensity (correct)
- Diminishing returns kick in quickly (correct)
- Optimal batch size ≠ maximum batch size (important insight)

**4. The Efficiency Puzzle:**
- T4 achieved 21% efficiency vs RTX 4080's 7%
- Even "compute-bound" workloads only achieve 7-36% of theoretical peak
- Framework overhead and thermal constraints matter more than I thought

### 6.2 Practical Takeaways

If I were to do this again or optimize these workloads, here's what I'd do:

1. **For my RTX 4080:** Use batch size 64, not 128. Monitor temperatures. Consider undervolting.

2. **For cloud training:** The T4 is actually a great choice - consistent performance and good efficiency.

3. **For all GPUs:** Try mixed precision (FP16) - could significantly improve both speed and memory usage.

4. **General advice:** Don't assume bigger/faster hardware always wins. Measure everything!

### 6.3 What I'd Do Differently

Looking back, here are the limitations:

1. **Dataset size** - ImageNette (10 classes) is smaller than full ImageNet. Results might differ at scale.

2. **CPU experiments** - I gave up on most CPU experiments because they were too slow. Would be interesting to see full results.

3. **Profiling tools** - I didn't use Nsight Compute for detailed profiling. This would have helped identify the exact bottlenecks.

4. **Mixed precision** - Only tested FP32. FP16 could change everything.

5. **Multiple runs** - I only did one run per configuration. Statistical significance testing would be better.
