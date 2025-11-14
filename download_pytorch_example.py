#!/usr/bin/env python3
"""
Download PyTorch ImageNet Example Script
Downloads the official PyTorch ImageNet training example.
"""

import urllib.request
from pathlib import Path

def download_pytorch_imagenet_example():
    """Download PyTorch ImageNet main.py example."""
    
    print("=" * 80)
    print("DOWNLOADING PYTORCH IMAGENET EXAMPLE")
    print("=" * 80)
    print()
    
    # URL to the main.py file in PyTorch examples repository
    url = "https://raw.githubusercontent.com/pytorch/examples/main/imagenet/main.py"
    
    # Destination
    project_root = Path(__file__).parent
    scripts_dir = project_root / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    destination = scripts_dir / "main.py"
    
    # Check if already exists
    if destination.exists():
        print(f"⚠️  File already exists: {destination}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return str(destination)
    
    # Download
    print(f"Downloading from: {url}")
    print(f"Destination: {destination}")
    print()
    
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Download complete!")
        print(f"  Saved to: {destination}")
        
        # Show file size
        size_kb = destination.stat().st_size / 1024
        print(f"  Size: {size_kb:.2f} KB")
        
        return str(destination)
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def main():
    """Main function."""
    result = download_pytorch_imagenet_example()
    
    if result:
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Review the downloaded main.py script")
        print("2. We'll modify it to add:")
        print("   - --max-iters argument")
        print("   - --warmup-iters argument")
        print("   - --log-file argument")
        print("   - Timing instrumentation")
        print("   - CSV logging")
        print("=" * 80)
    else:
        print("\n❌ Failed to download PyTorch ImageNet example")

if __name__ == "__main__":
    main()
