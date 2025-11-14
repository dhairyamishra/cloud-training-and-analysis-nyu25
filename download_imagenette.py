#!/usr/bin/env python3
"""
ImageNette Dataset Download Script
Downloads ImageNette dataset (10-class subset of ImageNet) for performance testing.
"""

import urllib.request
import tarfile
import os
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_imagenette():
    """Download and extract ImageNette dataset."""
    
    print("=" * 80)
    print("IMAGENETTE DATASET DOWNLOAD")
    print("=" * 80)
    print()
    
    # ImageNette URLs (full size version - 320px)
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    download_path = data_dir / "imagenette2-320.tgz"
    extract_dir = data_dir / "imagenette2-320"
    
    # Check if already downloaded
    if extract_dir.exists():
        print(f"✓ ImageNette already exists at: {extract_dir}")
        print("\nDataset structure:")
        print(f"  Train: {extract_dir / 'train'}")
        print(f"  Val: {extract_dir / 'val'}")
        return str(extract_dir)
    
    # Download
    print(f"Downloading ImageNette from: {url}")
    print(f"Destination: {download_path}")
    print()
    
    try:
        download_url(url, download_path)
        print(f"\n✓ Download complete: {download_path}")
        print(f"  Size: {download_path.stat().st_size / (1024**3):.2f} GB")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return None
    
    # Extract
    print("\nExtracting archive...")
    try:
        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        print(f"✓ Extraction complete: {extract_dir}")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None
    
    # Clean up archive
    if download_path.exists():
        download_path.unlink()
        print("✓ Cleaned up archive file")
    
    # Verify structure
    print("\n" + "=" * 80)
    print("DATASET VERIFICATION")
    print("=" * 80)
    
    train_dir = extract_dir / "train"
    val_dir = extract_dir / "val"
    
    if train_dir.exists() and val_dir.exists():
        train_classes = list(train_dir.iterdir())
        val_classes = list(val_dir.iterdir())
        
        print(f"\n✓ Training directory: {train_dir}")
        print(f"  Classes: {len(train_classes)}")
        
        # Count images in first class as sample
        if train_classes:
            sample_class = train_classes[0]
            sample_images = list(sample_class.glob("*.JPEG"))
            print(f"  Sample class '{sample_class.name}': {len(sample_images)} images")
        
        print(f"\n✓ Validation directory: {val_dir}")
        print(f"  Classes: {len(val_classes)}")
        
        if val_classes:
            sample_class = val_classes[0]
            sample_images = list(sample_class.glob("*.JPEG"))
            print(f"  Sample class '{sample_class.name}': {len(sample_images)} images")
    else:
        print("❌ Expected train/val directories not found")
        return None
    
    print("\n" + "=" * 80)
    print("IMAGENETTE CLASSES")
    print("=" * 80)
    print("\nImageNette contains 10 classes:")
    classes = {
        "n01440764": "tench (fish)",
        "n02102040": "English springer (dog)",
        "n02979186": "cassette player",
        "n03000684": "chain saw",
        "n03028079": "church",
        "n03394916": "French horn",
        "n03417042": "garbage truck",
        "n03425413": "gas pump",
        "n03445777": "golf ball",
        "n03888257": "parachute"
    }
    for code, name in classes.items():
        print(f"  {code}: {name}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"\n✓ Dataset ready at: {extract_dir}")
    print("\nYou can now:")
    print("1. Use this dataset directly for experiments (it's already small enough)")
    print("2. Or create an even smaller subset if needed")
    print("=" * 80)
    
    return str(extract_dir)

def main():
    """Main function."""
    dataset_path = download_imagenette()
    
    if dataset_path:
        print(f"\n✓ SUCCESS: ImageNette dataset is ready!")
        print(f"   Path: {dataset_path}")
    else:
        print("\n❌ FAILED: Could not download ImageNette dataset")

if __name__ == "__main__":
    main()
