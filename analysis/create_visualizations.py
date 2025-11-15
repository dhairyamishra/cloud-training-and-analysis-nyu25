#!/usr/bin/env python3
"""
Create Performance Visualizations
===================================

This script creates:
1. Bar charts for throughput comparison
2. Bar charts for time per iteration
3. Line plot for batch size effects
4. Grouped comparisons across environments

Usage:
    python analysis/create_visualizations.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_data():
    """Load aggregated results."""
    output_dir = Path('analysis/output')
    
    summary_df = pd.read_csv(output_dir / 'summary_statistics.csv')
    throughput_df = pd.read_csv(output_dir / 'throughput_comparison.csv')
    time_df = pd.read_csv(output_dir / 'time_comparison.csv')
    speedup_df = pd.read_csv(output_dir / 'speedup_analysis.csv')
    
    return summary_df, throughput_df, time_df, speedup_df


def plot_throughput_comparison(summary_df):
    """Create bar chart comparing throughput across configurations."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Filter to batch 128 for fair comparison
    df_b128 = summary_df[summary_df['batch_size'] == 128].copy()
    
    # Create grouped bar chart
    x = np.arange(len(df_b128['model'].unique()))
    width = 0.25
    
    environments = df_b128['environment'].unique()
    colors = {'RTX 4080': '#2ecc71', 'T4 GPU': '#3498db', 'CPU': '#e74c3c'}
    
    for i, env in enumerate(environments):
        env_data = df_b128[df_b128['environment'] == env]
        values = []
        for model in df_b128['model'].unique():
            model_data = env_data[env_data['model'] == model]
            if len(model_data) > 0:
                values.append(model_data['avg_throughput'].values[0])
            else:
                values.append(0)
        
        ax.bar(x + i*width, values, width, label=env, color=colors.get(env, '#95a5a6'))
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (images/sec)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Comparison Across Environments (Batch Size 128)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(df_b128['model'].unique())
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/output/throughput_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: throughput_comparison.png")
    plt.close()


def plot_time_comparison(summary_df):
    """Create bar chart comparing time per iteration."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Filter to batch 128
    df_b128 = summary_df[summary_df['batch_size'] == 128].copy()
    
    x = np.arange(len(df_b128['model'].unique()))
    width = 0.25
    
    environments = df_b128['environment'].unique()
    colors = {'RTX 4080': '#2ecc71', 'T4 GPU': '#3498db', 'CPU': '#e74c3c'}
    
    for i, env in enumerate(environments):
        env_data = df_b128[df_b128['environment'] == env]
        values = []
        for model in df_b128['model'].unique():
            model_data = env_data[env_data['model'] == model]
            if len(model_data) > 0:
                values.append(model_data['avg_time_ms'].values[0])
            else:
                values.append(0)
        
        ax.bar(x + i*width, values, width, label=env, color=colors.get(env, '#95a5a6'))
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time per Iteration (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Time per Iteration Comparison (Batch Size 128)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(df_b128['model'].unique())
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Use log scale if values vary greatly
    if df_b128['avg_time_ms'].max() / df_b128['avg_time_ms'].min() > 10:
        ax.set_yscale('log')
        ax.set_ylabel('Time per Iteration (ms) [log scale]', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('analysis/output/time_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: time_comparison.png")
    plt.close()


def plot_batch_size_sweep(summary_df):
    """Create line plot showing batch size effects on ResNet50."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter ResNet50 data
    resnet50_data = summary_df[summary_df['model'] == 'resnet50'].copy()
    
    environments = resnet50_data['environment'].unique()
    colors = {'RTX 4080': '#2ecc71', 'T4 GPU': '#3498db', 'CPU': '#e74c3c'}
    markers = {'RTX 4080': 'o', 'T4 GPU': 's', 'CPU': '^'}
    
    # Plot 1: Throughput vs Batch Size
    for env in environments:
        env_data = resnet50_data[resnet50_data['environment'] == env].sort_values('batch_size')
        ax1.plot(env_data['batch_size'], env_data['avg_throughput'], 
                marker=markers.get(env, 'o'), linewidth=2, markersize=8,
                label=env, color=colors.get(env, '#95a5a6'))
    
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (images/sec)', fontsize=12, fontweight='bold')
    ax1.set_title('ResNet50: Throughput vs Batch Size', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([32, 64, 128])
    
    # Plot 2: Time vs Batch Size
    for env in environments:
        env_data = resnet50_data[resnet50_data['environment'] == env].sort_values('batch_size')
        ax2.plot(env_data['batch_size'], env_data['avg_time_ms'], 
                marker=markers.get(env, 'o'), linewidth=2, markersize=8,
                label=env, color=colors.get(env, '#95a5a6'))
    
    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time per Iteration (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('ResNet50: Time per Iteration vs Batch Size', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([32, 64, 128])
    
    plt.tight_layout()
    plt.savefig('analysis/output/batch_size_sweep.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: batch_size_sweep.png")
    plt.close()


def plot_speedup_analysis(speedup_df):
    """Create bar chart showing speedup ratios."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    df = speedup_df.copy()
    df['config'] = df['model'] + '\n(b' + df['batch_size'].astype(str) + ')'
    
    x = np.arange(len(df))
    width = 0.35
    
    # Plot RTX 4080 vs T4 speedups
    if 'RTX4080_vs_T4' in df.columns:
        values = df['RTX4080_vs_T4'].fillna(0)
        ax.bar(x, values, width, label='RTX 4080 vs T4', color='#2ecc71')
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Equal Performance')
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup Ratio', fontsize=12, fontweight='bold')
    ax.set_title('RTX 4080 vs T4 GPU Speedup', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['config'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/output/speedup_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: speedup_analysis.png")
    plt.close()


def plot_model_comparison(summary_df):
    """Create grouped comparison for same model across environments."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = summary_df['model'].unique()
    colors = {'RTX 4080': '#2ecc71', 'T4 GPU': '#3498db', 'CPU': '#e74c3c'}
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = summary_df[summary_df['model'] == model]
        
        # Group by environment
        for env in model_data['environment'].unique():
            env_data = model_data[model_data['environment'] == env].sort_values('batch_size')
            ax.bar(env_data['batch_size'].astype(str), env_data['avg_throughput'],
                  label=env, color=colors.get(env, '#95a5a6'), alpha=0.8)
        
        ax.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
        ax.set_ylabel('Throughput (images/sec)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Model Performance Across Environments', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('analysis/output/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    plt.close()


def main():
    """Main visualization function."""
    
    print("=" * 80)
    print("CREATING PERFORMANCE VISUALIZATIONS")
    print("=" * 80)
    
    # Load data
    summary_df, throughput_df, time_df, speedup_df = load_data()
    
    # Create output directory
    output_dir = Path('analysis/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_throughput_comparison(summary_df)
    plot_time_comparison(summary_df)
    plot_batch_size_sweep(summary_df)
    plot_speedup_analysis(speedup_df)
    plot_model_comparison(summary_df)
    
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS CREATED")
    print("=" * 80)
    print("Saved 5 plots to: analysis/output/")
    print("  1. throughput_comparison.png")
    print("  2. time_comparison.png")
    print("  3. batch_size_sweep.png")
    print("  4. speedup_analysis.png")
    print("  5. model_comparison.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
