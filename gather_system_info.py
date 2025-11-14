#!/usr/bin/env python3
"""
System Information Gathering Script
Collects all necessary system specifications for the project documentation.
"""

import sys
import platform
import subprocess
import json
from datetime import datetime

def get_python_version():
    """Get Python version."""
    return sys.version

def get_os_info():
    """Get operating system information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor()
    }

def get_cuda_version():
    """Get CUDA version if available."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse CUDA version from output
            output = result.stdout
            for line in output.split('\n'):
                if 'release' in line.lower():
                    return line.strip()
        return "CUDA not found or nvcc not in PATH"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "CUDA not found or nvcc not in PATH"

def get_pytorch_info():
    """Get PyTorch and related package versions."""
    try:
        import torch
        import torchvision
        
        info = {
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version_pytorch": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A",
            "cudnn_enabled": torch.backends.cudnn.enabled if torch.cuda.is_available() else False,
        }
        return info
    except ImportError as e:
        return {"error": f"PyTorch not installed: {str(e)}"}

def get_gpu_info():
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap,driver_version', 
                               '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        gpus.append({
                            "name": parts[0],
                            "memory": parts[1],
                            "compute_capability": parts[2],
                            "driver_version": parts[3]
                        })
            return gpus
        return ["nvidia-smi command failed"]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ["nvidia-smi not found - GPU information unavailable"]

def get_cpu_info():
    """Get CPU information."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(['wmic', 'cpu', 'get', 'name'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return lines[1].strip()
        elif platform.system() == "Linux":
            result = subprocess.run(['lscpu'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Model name' in line:
                        return line.split(':')[1].strip()
        return platform.processor()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return platform.processor()

def get_installed_packages():
    """Get versions of key packages."""
    packages = {}
    package_list = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'fvcore', 'thop']
    
    for package in package_list:
        try:
            mod = __import__(package)
            packages[package] = getattr(mod, '__version__', 'version unknown')
        except ImportError:
            packages[package] = 'not installed'
    
    return packages

def get_gpu_specs():
    """Get detailed GPU specifications for RTX 4080 and T4."""
    gpu_specs = {
        "RTX_4080_Laptop": {
            "cuda_cores": "7424",
            "tensor_cores": "232 (4th gen)",
            "memory": "12 GB GDDR6",
            "memory_bandwidth": "432 GB/s",
            "base_clock": "~1350 MHz",
            "boost_clock": "~2280 MHz",
            "peak_fp32_tflops": "~22.6 TFLOPs",
            "peak_fp16_tflops": "~45.2 TFLOPs (with Tensor Cores)",
            "tdp": "~150W",
            "architecture": "Ada Lovelace (NVIDIA Ampere successor)"
        },
        "T4": {
            "cuda_cores": "2560",
            "tensor_cores": "320 (Turing)",
            "memory": "16 GB GDDR6",
            "memory_bandwidth": "320 GB/s",
            "base_clock": "585 MHz",
            "boost_clock": "1590 MHz",
            "peak_fp32_tflops": "8.1 TFLOPs",
            "peak_fp16_tflops": "65 TFLOPs (with Tensor Cores)",
            "tdp": "70W",
            "architecture": "Turing"
        }
    }
    return gpu_specs

def format_output(data):
    """Format the collected data for display."""
    output = []
    output.append("=" * 80)
    output.append("SYSTEM SPECIFICATIONS REPORT")
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 80)
    output.append("")
    
    # Python Version
    output.append("PYTHON VERSION:")
    output.append(f"  {data['python_version']}")
    output.append("")
    
    # Operating System
    output.append("OPERATING SYSTEM:")
    for key, value in data['os_info'].items():
        output.append(f"  {key.capitalize()}: {value}")
    output.append("")
    
    # CPU
    output.append("CPU:")
    output.append(f"  {data['cpu_info']}")
    output.append("")
    
    # CUDA
    output.append("CUDA:")
    output.append(f"  {data['cuda_version']}")
    output.append("")
    
    # PyTorch
    output.append("PYTORCH INFORMATION:")
    if 'error' in data['pytorch_info']:
        output.append(f"  {data['pytorch_info']['error']}")
    else:
        for key, value in data['pytorch_info'].items():
            output.append(f"  {key}: {value}")
    output.append("")
    
    # GPU
    output.append("GPU INFORMATION (Detected):")
    if isinstance(data['gpu_info'], list):
        for i, gpu in enumerate(data['gpu_info'], 1):
            if isinstance(gpu, dict):
                output.append(f"  GPU {i}:")
                for key, value in gpu.items():
                    output.append(f"    {key}: {value}")
            else:
                output.append(f"  {gpu}")
    output.append("")
    
    # Theoretical GPU Specs
    output.append("THEORETICAL GPU SPECIFICATIONS:")
    output.append("")
    for gpu_name, specs in data['gpu_specs'].items():
        output.append(f"  {gpu_name.replace('_', ' ')}:")
        for key, value in specs.items():
            output.append(f"    {key}: {value}")
        output.append("")
    
    # Installed Packages
    output.append("KEY PACKAGES:")
    for package, version in data['installed_packages'].items():
        output.append(f"  {package}: {version}")
    output.append("")
    
    output.append("=" * 80)
    
    return "\n".join(output)

def main():
    """Main function to gather and display system information."""
    print("Gathering system information...")
    print()
    
    data = {
        "python_version": get_python_version(),
        "os_info": get_os_info(),
        "cpu_info": get_cpu_info(),
        "cuda_version": get_cuda_version(),
        "pytorch_info": get_pytorch_info(),
        "gpu_info": get_gpu_info(),
        "gpu_specs": get_gpu_specs(),
        "installed_packages": get_installed_packages(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Format and print output
    formatted_output = format_output(data)
    print(formatted_output)
    
    # Save to file
    output_file = "system_specifications.txt"
    with open(output_file, 'w') as f:
        f.write(formatted_output)
    print(f"\nSystem specifications saved to: {output_file}")
    
    # Save JSON version
    json_file = "system_specifications.json"
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON version saved to: {json_file}")

if __name__ == "__main__":
    main()
