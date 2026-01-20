"""
DataLoader creation utilities.
"""
from typing import Tuple
from torch.utils.data import DataLoader

from src.data.dataset import FoodDataset
from src.data.transforms import get_train_transforms, get_val_transforms


def create_dataloaders(
    data_root: str,
    classes: list,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_root: Root directory containing data
        classes: List of class names to use
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        image_size: Image size for transforms
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    # Create datasets
    train_dataset = FoodDataset(
        root_dir=data_root,
        split="training",
        transform=train_transform,
        classes=classes,
    )
    
    val_dataset = FoodDataset(
        root_dir=data_root,
        split="validation",
        transform=val_transform,
        classes=classes,
    )
    
    test_dataset = FoodDataset(
        root_dir=data_root,
        split="evaluation",
        transform=val_transform,
        classes=classes,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
