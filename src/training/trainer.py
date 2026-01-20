"""
Training loop with W&B integration.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Optional

from src.training.metrics import calculate_metrics


class Trainer:
    """
    Trainer class with W&B logging.
    
    Handles:
    - Training loop
    - Validation
    - Metric logging to W&B
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        use_wandb: bool = True,
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            use_wandb: Whether to log to W&B
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_wandb = use_wandb
        
        self.current_epoch = 0
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> dict:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, targets, loss)
            
            # Accumulate
            total_loss += metrics["loss"] * images.size(0)
            total_correct += (metrics["accuracy"] / 100.0) * images.size(0)
            total_samples += images.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics['accuracy']:.2f}%"
            })
            
            # Log to W&B (every 10 batches)
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_loss": metrics["loss"],
                    "train/batch_acc": metrics["accuracy"],
                    "train/epoch": self.current_epoch,
                })
        
        # Epoch metrics
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        
        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
        }
    
    def validate(self) -> dict:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        
        with torch.no_grad():
            for images, targets in pbar:
                # Move to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, targets, loss)
                
                # Accumulate
                total_loss += metrics["loss"] * images.size(0)
                total_correct += (metrics["accuracy"] / 100.0) * images.size(0)
                total_samples += images.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']:.2f}%"
                })
        
        # Epoch metrics
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        
        return {
            "loss": avg_loss,
            "accuracy": avg_acc,
        }
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save best model (optional)
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    "train/epoch_loss": train_metrics["loss"],
                    "train/epoch_acc": train_metrics["accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/acc": val_metrics["accuracy"],
                    "epoch": epoch,
                })
            
            # Save best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                
                if save_path:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_acc": val_metrics["accuracy"],
                    }, save_path)
                    
                    print(f"  ✓ Saved best model (val_acc: {val_metrics['accuracy']:.2f}%)")
                    
                    # Log model to W&B
                    if self.use_wandb:
                        wandb.save(save_path)
        
        print("\n" + "=" * 60)
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print("=" * 60)
