#!/usr/bin/env python3
"""
Aggregate Experimental Results
================================

This script:
1. Loads all CSV files from results/ directories
2. Combines them into a single dataframe
3. Calculates summary statistics
4. Creates performance comparison tables

Usage:
    python analysis/aggregate_results.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_all_results(results_dir='results'):
    """Load all CSV files from results directories."""
    results_path = Path(results_dir)
    
    all_data = []
    csv_files = list(results_path.rglob('*.csv'))
    
    print(f"Found {len(csv_files)} CSV files")
    print("=" * 80)
    
    for csv_file in sorted(csv_files):
        # Skip test files
        if 'test' in csv_file.name.lower():
            continue
            
        print(f"Loading: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Add metadata from filename and path
            env_name = csv_file.parent.name  # e.g., envA_4080, envC_t4
            
            # Extract environment info
            if 'envA' in env_name or '4080' in env_name:
                df['environment'] = 'RTX 4080'
                df['env_code'] = 'A'
            elif 'envC' in env_name or 't4' in env_name.lower():
                df['environment'] = 'T4 GPU'
                df['env_code'] = 'C'
            elif 'envB' in env_name or 'cpu' in env_name.lower():
                df['environment'] = 'CPU'
                df['env_code'] = 'B'
            else:
                df['environment'] = 'Unknown'
                df['env_code'] = 'X'
            
            # Add source file
            df['source_file'] = csv_file.name
            
            all_data.append(df)
            print(f"  Loaded {len(df)} rows")
            
        except Exception as e:
            print(f"  ERROR loading {csv_file}: {e}")
    
    print("=" * 80)
    
    if not all_data:
        print("ERROR: No data loaded!")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Environments: {combined_df['environment'].unique()}")
    print(f"Models: {combined_df['model'].unique()}")
    print(f"Batch sizes: {sorted(combined_df['batch_size'].unique())}")
    
    return combined_df


def calculate_summary_statistics(df):
    """Calculate summary statistics for each configuration."""
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Group by environment, model, and batch size
    grouped = df.groupby(['environment', 'model', 'batch_size'])
    
    summary = grouped.agg({
        'iter_time_ms': ['mean', 'std', 'min', 'max', 'count'],
        'imgs_per_sec': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Rename for clarity
    summary = summary.rename(columns={
        'iter_time_ms_mean': 'avg_time_ms',
        'iter_time_ms_std': 'std_time_ms',
        'iter_time_ms_min': 'min_time_ms',
        'iter_time_ms_max': 'max_time_ms',
        'iter_time_ms_count': 'num_iterations',
        'imgs_per_sec_mean': 'avg_throughput',
        'imgs_per_sec_std': 'std_throughput',
        'imgs_per_sec_min': 'min_throughput',
        'imgs_per_sec_max': 'max_throughput'
    })
    
    return summary


def create_performance_table(summary_df):
    """Create a clean performance comparison table."""
    
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    
    # Create pivot table for easy comparison
    perf_table = summary_df.pivot_table(
        index=['model', 'batch_size'],
        columns='environment',
        values='avg_throughput',
        aggfunc='first'
    ).round(2)
    
    print("\nThroughput (images/sec):")
    print(perf_table.to_string())
    
    # Time per iteration table
    time_table = summary_df.pivot_table(
        index=['model', 'batch_size'],
        columns='environment',
        values='avg_time_ms',
        aggfunc='first'
    ).round(2)
    
    print("\n\nTime per iteration (ms):")
    print(time_table.to_string())
    
    return perf_table, time_table


def calculate_speedups(summary_df):
    """Calculate speedup ratios between environments."""
    
    print("\n" + "=" * 80)
    print("SPEEDUP ANALYSIS")
    print("=" * 80)
    
    speedups = []
    
    for (model, batch_size), group in summary_df.groupby(['model', 'batch_size']):
        env_data = dict(zip(group['environment'], group['avg_throughput']))
        
        speedup_row = {
            'model': model,
            'batch_size': batch_size
        }
        
        # RTX 4080 vs T4
        if 'RTX 4080' in env_data and 'T4 GPU' in env_data:
            speedup_row['RTX4080_vs_T4'] = round(env_data['RTX 4080'] / env_data['T4 GPU'], 2)
        
        # RTX 4080 vs CPU
        if 'RTX 4080' in env_data and 'CPU' in env_data:
            speedup_row['RTX4080_vs_CPU'] = round(env_data['RTX 4080'] / env_data['CPU'], 2)
        
        # T4 vs CPU
        if 'T4 GPU' in env_data and 'CPU' in env_data:
            speedup_row['T4_vs_CPU'] = round(env_data['T4 GPU'] / env_data['CPU'], 2)
        
        speedups.append(speedup_row)
    
    speedup_df = pd.DataFrame(speedups)
    print("\nSpeedup Ratios:")
    print(speedup_df.to_string(index=False))
    
    return speedup_df


def main():
    """Main analysis function."""
    
    print("=" * 80)
    print("EXPERIMENTAL RESULTS AGGREGATION")
    print("=" * 80)
    
    # Load all results
    df = load_all_results()
    
    if df is None:
        sys.exit(1)
    
    # Calculate summary statistics
    summary_df = calculate_summary_statistics(df)
    
    # Display full summary
    print("\n\nDetailed Summary:")
    print(summary_df.to_string(index=False))
    
    # Create performance tables
    perf_table, time_table = create_performance_table(summary_df)
    
    # Calculate speedups
    speedup_df = calculate_speedups(summary_df)
    
    # Save results
    output_dir = Path('analysis/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'combined_results.csv', index=False)
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    perf_table.to_csv(output_dir / 'throughput_comparison.csv')
    time_table.to_csv(output_dir / 'time_comparison.csv')
    speedup_df.to_csv(output_dir / 'speedup_analysis.csv', index=False)
    
    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"Combined data: {output_dir / 'combined_results.csv'}")
    print(f"Summary stats: {output_dir / 'summary_statistics.csv'}")
    print(f"Throughput table: {output_dir / 'throughput_comparison.csv'}")
    print(f"Time table: {output_dir / 'time_comparison.csv'}")
    print(f"Speedup analysis: {output_dir / 'speedup_analysis.csv'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
