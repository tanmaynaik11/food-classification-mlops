"""
Image classification model using pre-trained ResNet.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class FoodClassifier(nn.Module):
    """
    Food classification model based on ResNet18.
    
    Uses transfer learning:
    - Pre-trained ResNet18 backbone (ImageNet weights)
    - Replace final layer for our classes
    - Can freeze/unfreeze backbone
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pre-trained weights
            freeze_backbone: Freeze backbone weights (only train classifier)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained ResNet18
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get number of features from backbone
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: Optional[str] = None,
) -> FoodClassifier:
    """
    Create and initialize model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use pre-trained weights
        freeze_backbone: Freeze backbone weights
        device: Device to move model to
        
    Returns:
        Initialized model
    """
    model = FoodClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    
    trainable_params = model.get_num_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model
