"""
Training metrics calculation.
"""
import torch
from typing import Tuple


def calculate_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        outputs: Model outputs (logits) [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        
    Returns:
        accuracy: Accuracy as percentage (0-100)
    """
    _, predictions = torch.max(outputs, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = 100.0 * correct / total
    return accuracy


def calculate_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
) -> dict:
    """
    Calculate all metrics for logging.
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
        loss: Loss value
        
    Returns:
        Dictionary of metrics
    """
    accuracy = calculate_accuracy(outputs, targets)
    
    return {
        "loss": loss.item(),
        "accuracy": accuracy,
    }
