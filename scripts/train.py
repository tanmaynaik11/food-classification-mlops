"""
Main training script.

Usage:
    python scripts/train.py                    # Use default config
    python scripts/train.py --config quick_test  # Use quick_test config
"""
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn as nn
import wandb
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import create_dataloaders
from src.models.classifier import create_model
from src.training.trainer import Trainer


def load_config(config_name: str = "default") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "configs" / "training" / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_wandb(config: dict, use_wandb: bool = True):
    """Initialize W&B logging."""
    if not use_wandb:
        return
    
    # Generate run name if not provided
    run_name = config["wandb"].get("name")
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    # Initialize W&B
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=run_name,
        tags=config["wandb"]["tags"],
        config=config,
    )
    
    print(f"W&B initialized: {wandb.run.url}")


def main(config_name: str = "default", use_wandb: bool = True):
    """Main training function."""
    print("=" * 60)
    print("Food Classification Training")
    print("=" * 60)
    
    # Load config
    print(f"\nLoading config: {config_name}")
    config = load_config(config_name)
    
    # Setup W&B
    if use_wandb:
        setup_wandb(config, use_wandb)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config["data"]["root_dir"],
        classes=config["data"]["classes"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
        device=device,
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["optimizer"]["betas"],
    )
    
    # Create save directory
    save_dir = Path(config["paths"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Model save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"best_model_{timestamp}.pt"
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_wandb=use_wandb,
    )
    
    # Train
    trainer.train(
        num_epochs=config["training"]["num_epochs"],
        save_path=str(save_path),
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    # (We'll add this in next phase)
    
    # Finish W&B
    if use_wandb:
        wandb.finish()
    
    print(f"\nModel saved to: {save_path}")
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train food classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Config name (without .yaml extension)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    
    args = parser.parse_args()
    
    main(
        config_name=args.config,
        use_wandb=not args.no_wandb,
    )
