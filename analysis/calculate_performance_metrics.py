#!/usr/bin/env python3
"""
Calculate Performance Metrics
===============================

This script calculates:
1. Achieved FLOPs/s for each configuration
2. Theoretical vs achieved performance
3. Arithmetic intensity
4. Memory bandwidth utilization

Usage:
    python analysis/calculate_performance_metrics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# GPU Specifications
GPU_SPECS = {
    'RTX 4080': {
        'peak_fp32_tflops': 48.7,  # TFLOPs/s (laptop version)
        'memory_bandwidth_gbps': 384,  # GB/s (GDDR6X)
        'memory_size_gb': 12
    },
    'T4 GPU': {
        'peak_fp32_tflops': 8.1,  # TFLOPs/s
        'memory_bandwidth_gbps': 300,  # GB/s (GDDR6)
        'memory_size_gb': 16
    },
    'CPU': {
        'peak_fp32_tflops': 1.5,  # Estimated for i9-14900HX (24 cores)
        'memory_bandwidth_gbps': 89.6,  # DDR5-5600 dual channel
        'memory_size_gb': 32  # Typical
    }
}

def load_complexity_data():
    """Load model complexity data from Phase 3."""
    
    # Load complexity CSVs
    model_complexity = pd.read_csv('analysis/model_complexity.csv')
    batch_complexity = pd.read_csv('analysis/batch_complexity.csv')
    arithmetic_intensity = pd.read_csv('analysis/arithmetic_intensity.csv')
    
    # Normalize model names to lowercase
    batch_complexity['model'] = batch_complexity['model'].str.lower()
    arithmetic_intensity['model'] = arithmetic_intensity['model'].str.lower()
    
    return model_complexity, batch_complexity, arithmetic_intensity


def load_experimental_results():
    """Load aggregated experimental results."""
    
    summary_df = pd.read_csv('analysis/output/summary_statistics.csv')
    
    # Normalize model names
    summary_df['model'] = summary_df['model'].str.lower()
    
    return summary_df


def calculate_achieved_flops(summary_df, batch_complexity):
    """Calculate achieved FLOPs/s for each configuration."""
    
    print("=" * 80)
    print("CALCULATING ACHIEVED FLOPs/s")
    print("=" * 80)
    
    results = []
    
    for _, row in summary_df.iterrows():
        env = row['environment']
        model = row['model']
        batch_size = row['batch_size']
        avg_time_ms = row['avg_time_ms']
        
        # Get FLOPs per iteration from complexity data
        complexity_row = batch_complexity[
            (batch_complexity['model'] == model) & 
            (batch_complexity['batch_size'] == batch_size)
        ]
        
        if len(complexity_row) == 0:
            print(f"WARNING: No complexity data for {model} batch {batch_size}")
            continue
        
        # Use correct column name
        flops_per_iter = complexity_row['flops_per_iteration'].values[0]
        
        # Calculate achieved FLOPs/s
        # FLOPs/s = FLOPs per iteration / time per iteration (in seconds)
        time_sec = avg_time_ms / 1000.0
        achieved_flops_per_sec = flops_per_iter / time_sec
        achieved_tflops_per_sec = achieved_flops_per_sec / 1e12
        
        # Get theoretical peak
        if env in GPU_SPECS:
            peak_tflops = GPU_SPECS[env]['peak_fp32_tflops']
            efficiency = (achieved_tflops_per_sec / peak_tflops) * 100
        else:
            peak_tflops = np.nan
            efficiency = np.nan
        
        results.append({
            'environment': env,
            'model': model,
            'batch_size': batch_size,
            'flops_per_iter': flops_per_iter,
            'time_ms': avg_time_ms,
            'achieved_tflops_per_sec': round(achieved_tflops_per_sec, 3),
            'peak_tflops_per_sec': peak_tflops,
            'efficiency_percent': round(efficiency, 2) if not np.isnan(efficiency) else np.nan
        })
    
    results_df = pd.DataFrame(results)
    
    print("\nAchieved Performance:")
    print(results_df.to_string(index=False))
    
    return results_df


def calculate_arithmetic_intensity_actual(summary_df, arithmetic_intensity_df):
    """Calculate actual arithmetic intensity based on measurements."""
    
    print("\n" + "=" * 80)
    print("ARITHMETIC INTENSITY ANALYSIS")
    print("=" * 80)
    
    results = []
    
    for _, row in summary_df.iterrows():
        model = row['model']
        batch_size = row['batch_size']
        
        # Get theoretical arithmetic intensity
        ai_row = arithmetic_intensity_df[
            (arithmetic_intensity_df['model'] == model) & 
            (arithmetic_intensity_df['batch_size'] == batch_size)
        ]
        
        if len(ai_row) == 0:
            continue
        
        # Use correct column name
        arithmetic_intensity = ai_row['arithmetic_intensity'].values[0]
        
        results.append({
            'environment': row['environment'],
            'model': model,
            'batch_size': batch_size,
            'arithmetic_intensity': round(arithmetic_intensity, 2),
            'throughput_imgs_per_sec': row['avg_throughput']
        })
    
    results_df = pd.DataFrame(results)
    
    print("\nArithmetic Intensity by Configuration:")
    print(results_df.to_string(index=False))
    
    return results_df


def analyze_roofline_position(perf_df, ai_df):
    """Determine if each configuration is compute-bound or memory-bound."""
    
    print("\n" + "=" * 80)
    print("ROOFLINE ANALYSIS")
    print("=" * 80)
    
    # Merge performance and arithmetic intensity
    merged = perf_df.merge(
        ai_df[['environment', 'model', 'batch_size', 'arithmetic_intensity']], 
        on=['environment', 'model', 'batch_size']
    )
    
    results = []
    
    for _, row in merged.iterrows():
        env = row['environment']
        
        if env not in GPU_SPECS:
            continue
        
        peak_tflops = GPU_SPECS[env]['peak_fp32_tflops']
        peak_bandwidth_gbps = GPU_SPECS[env]['memory_bandwidth_gbps']
        
        # Ridge point: arithmetic intensity where compute and memory bounds meet
        # Ridge AI = Peak FLOPs/s / Peak Bandwidth
        # Convert: TFLOPs/s = 1e12 FLOPs/s, GB/s = 1e9 bytes/s
        ridge_ai = (peak_tflops * 1e12) / (peak_bandwidth_gbps * 1e9)
        
        ai = row['arithmetic_intensity']
        achieved_tflops = row['achieved_tflops_per_sec']
        
        # Determine if compute-bound or memory-bound
        if ai > ridge_ai:
            bound_type = 'Compute-bound'
            # Memory roofline: Performance = Bandwidth * AI
            memory_limited_tflops = (peak_bandwidth_gbps * 1e9 * ai) / 1e12
            limiting_factor = 'Compute'
        else:
            bound_type = 'Memory-bound'
            memory_limited_tflops = (peak_bandwidth_gbps * 1e9 * ai) / 1e12
            limiting_factor = 'Memory Bandwidth'
        
        results.append({
            'environment': env,
            'model': row['model'],
            'batch_size': row['batch_size'],
            'arithmetic_intensity': round(ai, 2),
            'ridge_point': round(ridge_ai, 2),
            'bound_type': bound_type,
            'achieved_tflops': round(achieved_tflops, 3),
            'peak_tflops': peak_tflops,
            'efficiency_percent': row['efficiency_percent']
        })
    
    results_df = pd.DataFrame(results)
    
    print("\nRoofline Position Analysis:")
    print(results_df.to_string(index=False))
    
    return results_df


def create_efficiency_summary(perf_df):
    """Create summary of efficiency across configurations."""
    
    print("\n" + "=" * 80)
    print("EFFICIENCY SUMMARY")
    print("=" * 80)
    
    # Group by environment
    for env in perf_df['environment'].unique():
        env_data = perf_df[perf_df['environment'] == env]
        
        print(f"\n{env}:")
        print(f"  Average efficiency: {env_data['efficiency_percent'].mean():.2f}%")
        print(f"  Max efficiency: {env_data['efficiency_percent'].max():.2f}%")
        print(f"  Min efficiency: {env_data['efficiency_percent'].min():.2f}%")
        
        # Best and worst configurations
        best = env_data.loc[env_data['efficiency_percent'].idxmax()]
        worst = env_data.loc[env_data['efficiency_percent'].idxmin()]
        
        print(f"  Best: {best['model']} (batch {best['batch_size']}) - {best['efficiency_percent']:.2f}%")
        print(f"  Worst: {worst['model']} (batch {worst['batch_size']}) - {worst['efficiency_percent']:.2f}%")


def main():
    """Main performance metrics calculation."""
    
    print("=" * 80)
    print("PERFORMANCE METRICS CALCULATION")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    model_complexity, batch_complexity, arithmetic_intensity_df = load_complexity_data()
    summary_df = load_experimental_results()
    
    # Calculate achieved FLOPs/s
    perf_df = calculate_achieved_flops(summary_df, batch_complexity)
    
    # Analyze arithmetic intensity
    ai_df = calculate_arithmetic_intensity_actual(summary_df, arithmetic_intensity_df)
    
    # Roofline analysis
    roofline_df = analyze_roofline_position(perf_df, ai_df)
    
    # Efficiency summary
    create_efficiency_summary(perf_df)
    
    # Save results
    output_dir = Path('analysis/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    perf_df.to_csv(output_dir / 'achieved_performance.csv', index=False)
    ai_df.to_csv(output_dir / 'arithmetic_intensity_analysis.csv', index=False)
    roofline_df.to_csv(output_dir / 'roofline_analysis.csv', index=False)
    
    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"Achieved performance: {output_dir / 'achieved_performance.csv'}")
    print(f"Arithmetic intensity: {output_dir / 'arithmetic_intensity_analysis.csv'}")
    print(f"Roofline analysis: {output_dir / 'roofline_analysis.csv'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
