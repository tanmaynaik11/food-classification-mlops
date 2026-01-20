"""
Test script to verify data loading works correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import create_dataloaders


def main():
    print("=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    # Configuration
    data_root = "data/raw"
    classes = ["Bread", "Dairy_product", "Dessert"]
    batch_size = 8
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        classes=classes,
        batch_size=batch_size,
        num_workers=0,  # 0 for testing (avoids multiprocessing issues)
        image_size=224,
    )
    
    print(f"\nDataloader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test loading one batch
    print("\nLoading one training batch...")
    images, labels = next(iter(train_loader))
    
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image dtype: {images.dtype}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Unique labels: {labels.unique().tolist()}")
    
    print("\n" + "=" * 60)
    print("✓ Data loading successful!")
    print("=" * 60)


if __name__ == "__main__":
    main()
