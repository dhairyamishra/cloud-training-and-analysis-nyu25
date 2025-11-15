#!/usr/bin/env python3
"""
Create Roofline Plots
======================

This script creates roofline model visualizations for:
1. RTX 4080 GPU
2. T4 GPU
3. CPU (optional)

Usage:
    python analysis/create_roofline_plots.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# GPU Specifications
GPU_SPECS = {
    'RTX 4080': {
        'peak_fp32_tflops': 48.7,
        'memory_bandwidth_gbps': 384,
        'color': '#2ecc71',
        'marker': 'o'
    },
    'T4 GPU': {
        'peak_fp32_tflops': 8.1,
        'memory_bandwidth_gbps': 300,
        'color': '#3498db',
        'marker': 's'
    },
    'CPU': {
        'peak_fp32_tflops': 1.5,
        'memory_bandwidth_gbps': 89.6,
        'color': '#e74c3c',
        'marker': '^'
    }
}

def load_data():
    """Load roofline analysis data."""
    roofline_df = pd.read_csv('analysis/output/roofline_analysis.csv')
    return roofline_df


def create_roofline_plot(env_name, roofline_df, save_path):
    """Create roofline plot for a specific environment."""
    
    if env_name not in GPU_SPECS:
        print(f"WARNING: No specs for {env_name}")
        return
    
    specs = GPU_SPECS[env_name]
    peak_tflops = specs['peak_fp32_tflops']
    bandwidth_gbps = specs['memory_bandwidth_gbps']
    
    # Calculate ridge point
    # Ridge AI = Peak FLOPs/s / Peak Bandwidth
    ridge_ai = (peak_tflops * 1e12) / (bandwidth_gbps * 1e9)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define arithmetic intensity range
    ai_range = np.logspace(-1, 3, 1000)  # 0.1 to 1000 FLOPs/byte
    
    # Memory-bound region (linear with AI)
    memory_bound = (bandwidth_gbps * 1e9 * ai_range) / 1e12  # Convert to TFLOPs/s
    
    # Compute-bound region (flat at peak)
    compute_bound = np.full_like(ai_range, peak_tflops)
    
    # Roofline is the minimum of the two
    roofline = np.minimum(memory_bound, compute_bound)
    
    # Plot roofline
    ax.loglog(ai_range, roofline, 'k-', linewidth=3, label='Roofline', zorder=1)
    
    # Shade regions
    ax.fill_between(ai_range, 0, roofline, alpha=0.1, color='gray')
    
    # Plot ridge point
    ax.axvline(x=ridge_ai, color='red', linestyle='--', linewidth=2, 
               label=f'Ridge Point (AI={ridge_ai:.1f})', zorder=2)
    
    # Add memory-bound and compute-bound labels
    ax.text(1, bandwidth_gbps * 1e9 * 1 / 1e12 * 0.5, 'Memory\nBound', 
            fontsize=14, fontweight='bold', color='darkred', 
            ha='center', va='center', rotation=45)
    ax.text(ridge_ai * 10, peak_tflops * 0.7, 'Compute\nBound', 
            fontsize=14, fontweight='bold', color='darkblue',
            ha='center', va='center')
    
    # Filter data for this environment
    env_data = roofline_df[roofline_df['environment'] == env_name]
    
    # Plot experimental points
    model_markers = {'resnet18': 'o', 'resnet50': 's', 'mobilenet_v2': '^'}
    model_colors = {'resnet18': '#e74c3c', 'resnet50': '#3498db', 'mobilenet_v2': '#2ecc71'}
    
    for model in env_data['model'].unique():
        model_data = env_data[env_data['model'] == model]
        
        ax.scatter(model_data['arithmetic_intensity'], 
                  model_data['achieved_tflops'],
                  s=200, marker=model_markers.get(model, 'o'),
                  color=model_colors.get(model, 'gray'),
                  edgecolors='black', linewidths=2,
                  label=f'{model.replace("_", " ").title()}',
                  zorder=3, alpha=0.8)
        
        # Add batch size labels
        for _, row in model_data.iterrows():
            ax.annotate(f'b{int(row["batch_size"])}', 
                       (row['arithmetic_intensity'], row['achieved_tflops']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (TFLOPs/s)', fontsize=14, fontweight='bold')
    ax.set_title(f'Roofline Model: {env_name}\n'
                f'Peak: {peak_tflops} TFLOPs/s | Bandwidth: {bandwidth_gbps} GB/s',
                fontsize=16, fontweight='bold', pad=20)
    
    # Set limits
    ax.set_xlim([10, 1000])
    ax.set_ylim([0.01, peak_tflops * 2])
    
    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Add efficiency lines
    for eff in [10, 25, 50, 75]:
        eff_performance = peak_tflops * (eff / 100)
        ax.axhline(y=eff_performance, color='gray', linestyle=':', 
                  linewidth=1, alpha=0.5)
        ax.text(ax.get_xlim()[1] * 0.95, eff_performance, f'{eff}%',
               fontsize=8, color='gray', ha='right', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_combined_roofline(roofline_df, save_path):
    """Create combined roofline plot comparing RTX 4080 and T4."""
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Define AI range
    ai_range = np.logspace(-1, 3, 1000)
    
    # Plot rooflines for both GPUs
    for env_name in ['RTX 4080', 'T4 GPU']:
        if env_name not in GPU_SPECS:
            continue
        
        specs = GPU_SPECS[env_name]
        peak_tflops = specs['peak_fp32_tflops']
        bandwidth_gbps = specs['memory_bandwidth_gbps']
        color = specs['color']
        
        # Calculate roofline
        memory_bound = (bandwidth_gbps * 1e9 * ai_range) / 1e12
        compute_bound = np.full_like(ai_range, peak_tflops)
        roofline = np.minimum(memory_bound, compute_bound)
        
        # Plot
        ax.loglog(ai_range, roofline, linewidth=3, 
                 label=f'{env_name} Roofline', color=color, alpha=0.7)
        
        # Ridge point
        ridge_ai = (peak_tflops * 1e12) / (bandwidth_gbps * 1e9)
        ax.axvline(x=ridge_ai, color=color, linestyle='--', 
                  linewidth=1.5, alpha=0.5)
    
    # Plot experimental points
    model_markers = {'resnet18': 'o', 'resnet50': 's', 'mobilenet_v2': '^'}
    
    for env_name in ['RTX 4080', 'T4 GPU']:
        env_data = roofline_df[roofline_df['environment'] == env_name]
        color = GPU_SPECS[env_name]['color']
        
        for model in env_data['model'].unique():
            model_data = env_data[env_data['model'] == model]
            
            ax.scatter(model_data['arithmetic_intensity'], 
                      model_data['achieved_tflops'],
                      s=150, marker=model_markers.get(model, 'o'),
                      color=color, edgecolors='black', linewidths=1.5,
                      label=f'{env_name} - {model.replace("_", " ").title()}',
                      alpha=0.7, zorder=3)
    
    # Formatting
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (TFLOPs/s)', fontsize=14, fontweight='bold')
    ax.set_title('Roofline Comparison: RTX 4080 vs T4 GPU',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim([10, 1000])
    ax.set_ylim([0.1, 60])
    
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Main roofline plotting function."""
    
    print("=" * 80)
    print("CREATING ROOFLINE PLOTS")
    print("=" * 80)
    
    # Load data
    roofline_df = load_data()
    
    # Create output directory
    output_dir = Path('analysis/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create individual roofline plots
    print("\nGenerating roofline plots...")
    
    for env in roofline_df['environment'].unique():
        if env in GPU_SPECS:
            save_path = output_dir / f'roofline_{env.lower().replace(" ", "_")}.png'
            create_roofline_plot(env, roofline_df, save_path)
    
    # Create combined plot
    save_path = output_dir / 'roofline_comparison.png'
    create_combined_roofline(roofline_df, save_path)
    
    print("\n" + "=" * 80)
    print("ALL ROOFLINE PLOTS CREATED")
    print("=" * 80)
    print("Saved plots to: analysis/output/")
    print("  - roofline_rtx_4080.png")
    print("  - roofline_t4_gpu.png")
    print("  - roofline_comparison.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
