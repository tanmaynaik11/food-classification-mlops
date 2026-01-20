"""
Download Food-11 dataset for training.
Small dataset (11 classes, ~16K images) perfect for quick experiments.
"""
import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, destination: Path) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=destination.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def main():
    """Download and extract Food-11 dataset."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs (using mirror since original can be slow)
    urls = {
        "training": "https://www.kaggle.com/api/v1/datasets/download/trolukovich/food11-image-dataset",
    }
    
    print("=" * 60)
    print("Food-11 Dataset Download")
    print("=" * 60)
    print(f"Download location: {raw_dir}")
    print()
    
    # Note: This is a simplified version
    # For Kaggle datasets, you need Kaggle API credentials
    # For now, we'll provide instructions for manual download
    
    print("MANUAL DOWNLOAD REQUIRED:")
    print()
    print("1. Go to: https://www.kaggle.com/datasets/trolukovich/food11-image-dataset")
    print("2. Click 'Download' (requires Kaggle account - free)")
    print("3. Extract the zip file")
    print(f"4. Move the extracted folders to: {raw_dir}/")
    print()
    print("Expected structure after extraction:")
    print(f"{raw_dir}/")
    print("  ├── training/")
    print("  ├── validation/")
    print("  └── evaluation/")
    print()
    print("Alternative: Use Food-11 from PyTorch (we'll add this option)")
    
    # Check if data already exists
    training_dir = raw_dir / "training"
    if training_dir.exists():
        num_images = len(list(training_dir.rglob("*.jpg")))
        print(f"✓ Found existing data: {num_images} training images")
    else:
        print("✗ No data found yet")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
