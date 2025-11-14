#!/usr/bin/env python3
"""
Project Directory Structure Setup Script
Creates all necessary directories for the ImageNet performance modeling project.
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create the complete project directory structure."""
    
    # Define the directory structure
    directories = [
        "scripts",           # Training scripts and utilities
        "data",             # Dataset storage
        "data/imagenet_subset",  # ImageNet subset
        "data/imagenet_subset/train",  # Training data
        "data/imagenet_subset/val",    # Validation data
        "results",          # Experiment logs and CSV files
        "results/envA_4080",  # RTX 4080 results
        "results/envB_cpu",   # CPU results
        "results/envC_t4",    # T4 GPU results
        "analysis",         # Analysis notebooks and scripts
        "analysis/notebooks",  # Jupyter notebooks
        "analysis/plots",     # Generated plots
        "roofline",         # Roofline data and plots
        "roofline/data",    # Roofline raw data
        "roofline/plots",   # Roofline plots
        "docs",             # Documentation and report
        "docs/figures",     # Figures for report
        "docs/tables",      # Tables for report
    ]
    
    # Get the project root directory (where this script is located)
    project_root = Path(__file__).parent
    
    print("=" * 80)
    print("PROJECT DIRECTORY STRUCTURE SETUP")
    print("=" * 80)
    print(f"\nProject Root: {project_root}")
    print("\nCreating directories...\n")
    
    created_dirs = []
    existing_dirs = []
    
    for directory in directories:
        dir_path = project_root / directory
        if dir_path.exists():
            existing_dirs.append(directory)
            print(f"  ✓ {directory} (already exists)")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
            print(f"  + {directory} (created)")
    
    # Create README files in key directories
    readme_content = {
        "scripts": "# Scripts Directory\n\nContains training scripts and utility functions.\n\n- `main.py`: Modified PyTorch ImageNet training script\n- `complexity_analysis.py`: FLOPs and complexity calculations\n- `create_imagenet_subset.py`: Dataset subset creation\n",
        
        "data": "# Data Directory\n\nContains the ImageNet subset dataset.\n\n- `imagenet_subset/train/`: Training images organized by class\n- `imagenet_subset/val/`: Validation images organized by class\n",
        
        "results": "# Results Directory\n\nContains experimental results and logs.\n\n- `envA_4080/`: RTX 4080 GPU results\n- `envB_cpu/`: CPU-only results\n- `envC_t4/`: Google Colab T4 GPU results\n\nEach CSV file follows the naming convention: `{env}_{device}_{model}_bs{batch_size}.csv`\n",
        
        "analysis": "# Analysis Directory\n\nContains analysis scripts, notebooks, and generated plots.\n\n- `notebooks/`: Jupyter notebooks for data analysis\n- `plots/`: Performance comparison plots\n",
        
        "roofline": "# Roofline Directory\n\nContains roofline modeling data and plots.\n\n- `data/`: Raw profiling data and metrics\n- `plots/`: Roofline model visualizations\n",
        
        "docs": "# Documentation Directory\n\nContains the final report and supporting materials.\n\n- `figures/`: All figures for the report\n- `tables/`: All tables for the report\n- Final report document\n",
    }
    
    print("\nCreating README files...\n")
    
    for dir_name, content in readme_content.items():
        readme_path = project_root / dir_name / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w') as f:
                f.write(content)
            print(f"  + {dir_name}/README.md (created)")
        else:
            print(f"  ✓ {dir_name}/README.md (already exists)")
    
    # Create .gitignore if it doesn't exist
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt

# Data (large files)
data/imagenet_subset/train/*
data/imagenet_subset/val/*
!data/imagenet_subset/train/.gitkeep
!data/imagenet_subset/val/.gitkeep

# Results (can be large)
results/**/*.csv
results/**/*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.bak
"""
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print(f"\n  + .gitignore (created)")
    else:
        print(f"\n  ✓ .gitignore (already exists)")
    
    # Create .gitkeep files in data directories to preserve structure
    gitkeep_dirs = [
        "data/imagenet_subset/train",
        "data/imagenet_subset/val",
    ]
    
    print("\nCreating .gitkeep files...\n")
    for dir_name in gitkeep_dirs:
        gitkeep_path = project_root / dir_name / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.touch()
            print(f"  + {dir_name}/.gitkeep (created)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nDirectories created: {len(created_dirs)}")
    print(f"Directories already existing: {len(existing_dirs)}")
    print(f"Total directories: {len(directories)}")
    print("\n✓ Project directory structure setup complete!")
    print("\nNext steps:")
    print("  1. Run gather_system_info.py to document system specifications")
    print("  2. Set up Python virtual environment")
    print("  3. Install required packages (PyTorch, torchvision, etc.)")
    print("=" * 80)

if __name__ == "__main__":
    create_directory_structure()
