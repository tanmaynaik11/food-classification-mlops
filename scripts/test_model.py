"""
Test script to verify model creation works.
"""
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classifier import create_model


def main():
    print("=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    # Create model
    num_classes = 3
    model = create_model(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=False,
    )
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Move to same device as model
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: [{batch_size}, {num_classes}]")
    
    # Verify output shape
    assert output.shape == (batch_size, num_classes), "Output shape mismatch!"
    
    print("\n" + "=" * 60)
    print("✓ Model creation successful!")
    print("=" * 60)


if __name__ == "__main__":
    main()
