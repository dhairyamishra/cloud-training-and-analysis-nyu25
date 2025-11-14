#!/usr/bin/env python3
"""
Model Complexity Analysis Script
Calculates FLOPs, parameters, and arithmetic intensity for CNN models.
"""

import torch
import torchvision.models as models
from fvcore.nn import FlopCountAnalysis, parameter_count
import pandas as pd
from pathlib import Path

def count_flops_fvcore(model, input_shape=(1, 3, 224, 224)):
    """Count FLOPs using fvcore."""
    dummy_input = torch.randn(input_shape)
    flops = FlopCountAnalysis(model, dummy_input)
    return flops.total()

def count_parameters(model):
    """Count total parameters."""
    return parameter_count(model)['']

def analyze_model(model_name, model, input_shape=(1, 3, 224, 224)):
    """Analyze a single model."""
    print(f"\nAnalyzing {model_name}...")
    
    # Set model to eval mode
    model.eval()
    
    # Count FLOPs (inference)
    flops_inference = count_flops_fvcore(model, input_shape)
    
    # Count parameters
    params = count_parameters(model)
    
    # Estimate training FLOPs (forward + backward ≈ 3x inference)
    flops_training = flops_inference * 3
    
    # Calculate model size in MB (assuming FP32)
    model_size_mb = (params * 4) / (1024 ** 2)
    
    results = {
        'model': model_name,
        'parameters': params,
        'parameters_millions': params / 1e6,
        'model_size_mb': model_size_mb,
        'flops_inference': flops_inference,
        'flops_inference_gflops': flops_inference / 1e9,
        'flops_training_per_image': flops_training,
        'flops_training_gflops': flops_training / 1e9,
    }
    
    print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
    print(f"  Model Size: {model_size_mb:.2f} MB")
    print(f"  FLOPs (inference): {flops_inference/1e9:.2f} GFLOPs")
    print(f"  FLOPs (training): {flops_training/1e9:.2f} GFLOPs")
    
    return results

def calculate_batch_complexities(model_results, batch_sizes=[32, 64, 128, 256]):
    """Calculate FLOPs per iteration for different batch sizes."""
    batch_data = []
    
    for result in model_results:
        model_name = result['model']
        flops_per_image = result['flops_training_per_image']
        
        for batch_size in batch_sizes:
            flops_per_iter = flops_per_image * batch_size
            
            batch_data.append({
                'model': model_name,
                'batch_size': batch_size,
                'flops_per_iteration': flops_per_iter,
                'flops_per_iteration_gflops': flops_per_iter / 1e9,
                'flops_per_iteration_tflops': flops_per_iter / 1e12,
            })
    
    return batch_data

def estimate_arithmetic_intensity(model_results, batch_sizes=[32, 64, 128, 256]):
    """
    Estimate arithmetic intensity (FLOPs/byte).
    
    Simplified calculation:
    - Memory traffic ≈ model parameters (weights) + activations
    - For training: need to load weights, store activations, gradients
    """
    ai_data = []
    
    for result in model_results:
        model_name = result['model']
        params = result['parameters']
        flops_per_image = result['flops_training_per_image']
        
        # Estimate memory traffic per image (bytes)
        # Weights: params * 4 bytes (FP32)
        # Activations: rough estimate based on model size
        # For simplicity: memory_traffic ≈ 2 * params * 4 (weights + gradients)
        weights_bytes = params * 4
        
        for batch_size in batch_sizes:
            flops_per_iter = flops_per_image * batch_size
            
            # Memory traffic per iteration
            # Weights are loaded once per iteration
            # Activations scale with batch size (rough estimate)
            activation_bytes_estimate = params * 4 * 0.5 * batch_size  # rough estimate
            memory_traffic = weights_bytes + activation_bytes_estimate
            
            # Arithmetic intensity (FLOPs/byte)
            arithmetic_intensity = flops_per_iter / memory_traffic
            
            ai_data.append({
                'model': model_name,
                'batch_size': batch_size,
                'memory_traffic_mb': memory_traffic / (1024 ** 2),
                'memory_traffic_gb': memory_traffic / (1024 ** 3),
                'arithmetic_intensity': arithmetic_intensity,
            })
    
    return ai_data

def main():
    """Main analysis function."""
    print("=" * 80)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 80)
    
    # Define models to analyze
    models_to_analyze = {
        'ResNet18': models.resnet18(pretrained=False),
        'ResNet50': models.resnet50(pretrained=False),
        'MobileNetV2': models.mobilenet_v2(pretrained=False),
    }
    
    # Analyze each model
    model_results = []
    for model_name, model in models_to_analyze.items():
        result = analyze_model(model_name, model)
        model_results.append(result)
    
    # Create DataFrame for model complexities
    df_models = pd.DataFrame(model_results)
    
    print("\n" + "=" * 80)
    print("MODEL COMPLEXITY SUMMARY")
    print("=" * 80)
    print(df_models.to_string(index=False))
    
    # Calculate batch complexities
    print("\n" + "=" * 80)
    print("BATCH SIZE ANALYSIS")
    print("=" * 80)
    
    batch_sizes = [32, 64, 128, 256]
    batch_data = calculate_batch_complexities(model_results, batch_sizes)
    df_batch = pd.DataFrame(batch_data)
    
    print("\nFLOPs per iteration for different batch sizes:")
    print(df_batch.to_string(index=False))
    
    # Calculate arithmetic intensity
    print("\n" + "=" * 80)
    print("ARITHMETIC INTENSITY ESTIMATES")
    print("=" * 80)
    
    ai_data = estimate_arithmetic_intensity(model_results, batch_sizes)
    df_ai = pd.DataFrame(ai_data)
    
    print("\nArithmetic Intensity (FLOPs/byte):")
    print(df_ai.to_string(index=False))
    
    # Save results
    output_dir = Path(__file__).parent.parent / "analysis"
    output_dir.mkdir(exist_ok=True)
    
    df_models.to_csv(output_dir / "model_complexity.csv", index=False)
    df_batch.to_csv(output_dir / "batch_complexity.csv", index=False)
    df_ai.to_csv(output_dir / "arithmetic_intensity.csv", index=False)
    
    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"  Model complexity: {output_dir / 'model_complexity.csv'}")
    print(f"  Batch complexity: {output_dir / 'batch_complexity.csv'}")
    print(f"  Arithmetic intensity: {output_dir / 'arithmetic_intensity.csv'}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("\n1. Model Comparison:")
    print(f"   - ResNet50 has ~{df_models.loc[1, 'parameters_millions']/df_models.loc[0, 'parameters_millions']:.1f}x more parameters than ResNet18")
    print(f"   - ResNet50 requires ~{df_models.loc[1, 'flops_training_gflops']/df_models.loc[0, 'flops_training_gflops']:.1f}x more FLOPs than ResNet18")
    print(f"   - MobileNetV2 is the most efficient with only {df_models.loc[2, 'parameters_millions']:.1f}M parameters")
    
    print("\n2. Batch Size Impact:")
    print(f"   - Doubling batch size doubles FLOPs per iteration")
    print(f"   - Batch 256 vs Batch 32: {256/32:.0f}x more FLOPs per iteration")
    
    print("\n3. Arithmetic Intensity:")
    print(f"   - Higher batch sizes generally increase arithmetic intensity")
    print(f"   - This helps utilize GPU compute capacity more efficiently")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
