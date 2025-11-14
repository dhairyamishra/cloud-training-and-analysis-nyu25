#!/usr/bin/env python3
"""
Kaggle API Setup Script
Sets up Kaggle API credentials for downloading ImageNet dataset.
"""

import os
import shutil
import json
from pathlib import Path

def setup_kaggle_credentials():
    """Set up Kaggle API credentials."""
    
    print("=" * 80)
    print("KAGGLE API SETUP")
    print("=" * 80)
    print()
    
    # Check if kaggle.json exists in project directory
    project_root = Path(__file__).parent
    kaggle_json_source = project_root / "kaggle.json"
    
    if not kaggle_json_source.exists():
        print("❌ Error: kaggle.json not found in project directory")
        print("Please download your Kaggle API token from:")
        print("https://www.kaggle.com/settings -> API -> Create New API Token")
        return False
    
    # Verify kaggle.json format
    try:
        with open(kaggle_json_source, 'r') as f:
            creds = json.load(f)
            if 'username' not in creds or 'key' not in creds:
                print("❌ Error: kaggle.json is missing 'username' or 'key'")
                return False
            print(f"✓ Found Kaggle credentials for user: {creds['username']}")
    except json.JSONDecodeError:
        print("❌ Error: kaggle.json is not valid JSON")
        return False
    
    # Determine Kaggle config directory based on OS
    if os.name == 'nt':  # Windows
        kaggle_dir = Path.home() / '.kaggle'
    else:  # Linux/Mac
        kaggle_dir = Path.home() / '.kaggle'
    
    # Create .kaggle directory if it doesn't exist
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy kaggle.json to .kaggle directory
    kaggle_json_dest = kaggle_dir / "kaggle.json"
    shutil.copy2(kaggle_json_source, kaggle_json_dest)
    
    # Set proper permissions (important for Linux/Mac)
    if os.name != 'nt':
        os.chmod(kaggle_json_dest, 0o600)
    
    print(f"✓ Kaggle credentials copied to: {kaggle_json_dest}")
    print()
    
    return True

def install_kaggle_package():
    """Check if kaggle package is installed."""
    try:
        import kaggle
        print("✓ Kaggle package is already installed")
        return True
    except ImportError:
        print("❌ Kaggle package not installed")
        print("\nTo install, run:")
        print("  pip install kaggle")
        return False

def test_kaggle_api():
    """Test Kaggle API connection."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("\n✓ Kaggle API authentication successful!")
        print("\nYour Kaggle datasets are accessible.")
        return True
        
    except Exception as e:
        print(f"\n❌ Kaggle API authentication failed: {e}")
        return False

def list_imagenet_datasets():
    """List available ImageNet datasets on Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("\n" + "=" * 80)
        print("SEARCHING FOR IMAGENET DATASETS ON KAGGLE")
        print("=" * 80)
        print()
        
        # Search for ImageNet datasets
        datasets = api.dataset_list(search='imagenet')
        
        if datasets:
            print(f"Found {len(datasets)} ImageNet-related datasets:\n")
            for i, dataset in enumerate(datasets[:10], 1):  # Show top 10
                print(f"{i}. {dataset.ref}")
                print(f"   Title: {dataset.title}")
                print(f"   Size: {dataset.size}")
                print(f"   Downloads: {dataset.downloadCount}")
                print()
        else:
            print("No ImageNet datasets found.")
            print("\nNote: Full ImageNet may require direct download from image-net.org")
            print("Alternative: Use ImageNette (smaller subset) for testing")
        
        return True
        
    except Exception as e:
        print(f"Error searching datasets: {e}")
        return False

def main():
    """Main setup function."""
    
    # Step 1: Set up credentials
    if not setup_kaggle_credentials():
        print("\n⚠️  Setup incomplete. Please fix the issues above.")
        return
    
    print()
    
    # Step 2: Check if kaggle package is installed
    if not install_kaggle_package():
        print("\n⚠️  Please install kaggle package first:")
        print("     pip install kaggle")
        return
    
    print()
    
    # Step 3: Test API connection
    if not test_kaggle_api():
        print("\n⚠️  API authentication failed. Please check your credentials.")
        return
    
    # Step 4: List available ImageNet datasets
    list_imagenet_datasets()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Choose an ImageNet dataset from the list above")
    print("2. Or use ImageNette for quick testing:")
    print("   kaggle datasets download -d fastai/imagenette")
    print()
    print("3. We'll create a subset extraction script next")
    print("=" * 80)

if __name__ == "__main__":
    main()
