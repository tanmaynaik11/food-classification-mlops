"""
PyTorch Dataset for Food Classification.
Handles loading images from folder structure.
"""
import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List

import torch
from torch.utils.data import Dataset
from PIL import Image


class FoodDataset(Dataset):
    """
    Food classification dataset.
    
    Expected folder structure:
        root_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                img4.jpg
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "training",
        transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None,
    ):
        """
        Args:
            root_dir: Root directory containing data (e.g., 'data/raw')
            split: Which split to use ('training', 'validation', 'evaluation')
            transform: Optional transform to apply to images
            classes: Optional list of class names to use (uses all if None)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Build path to split directory
        self.split_dir = self.root_dir / split
        
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")
        
        # Get class folders
        if classes is None:
            # Auto-detect classes from folders
            self.classes = sorted([
                d.name for d in self.split_dir.iterdir() 
                if d.is_dir() and not d.name.startswith('.')
            ])
        else:
            self.classes = classes
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} images from {split} split")
        print(f"Classes ({len(self.classes)}): {', '.join(self.classes)}")
    
    def _load_samples(self):
        """Load all image paths and corresponding labels."""
        for class_name in self.classes:
            class_dir = self.split_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Find all jpg images
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.split_dir}")
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Args:
            idx: Index of sample
            
        Returns:
            image: Transformed image tensor
            label: Class index
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> dict:
        """Get number of samples per class."""
        counts = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            counts[class_name] += 1
        return counts


def get_class_weights(dataset: FoodDataset) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        dataset: FoodDataset instance
        
    Returns:
        weights: Tensor of class weights (inverse frequency)
    """
    counts = dataset.get_class_counts()
    total = sum(counts.values())
    
    weights = []
    for cls in dataset.classes:
        weight = total / (len(dataset.classes) * counts[cls])
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)
