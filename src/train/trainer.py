"""Training utilities for physics simulation models."""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.physics_data import PhysicsDataset, create_synthetic_physics_data
from ..models.physics_gnn import PhysicsGNN
from ..utils.device import get_device, save_model, load_model


class PhysicsTrainer:
    """Trainer for physics simulation models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "checkpoints"
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            device: Device to train on.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader.
            criterion: Loss function.
            
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            if isinstance(batch, list):
                # Handle sequence data
                for i in range(len(batch) - 1):
                    current_data = batch[i].to(self.device)
                    next_data = batch[i + 1].to(self.device)
                    
                    loss = self._train_step(current_data, next_data, criterion)
                    total_loss += loss
                    num_batches += 1
            else:
                # Handle single graph data
                batch = batch.to(self.device)
                loss = self._train_step(batch, batch, criterion)
                total_loss += loss
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _train_step(
        self,
        current_data,
        next_data,
        criterion: nn.Module
    ) -> float:
        """Single training step.
        
        Args:
            current_data: Current time step data.
            next_data: Next time step data.
            criterion: Loss function.
            
        Returns:
            Loss value.
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_vel_update = self.model(
            current_data.x,
            current_data.edge_index,
            current_data.pos
        )
        
        # Compute target velocity update
        target_vel_update = next_data.x[:, :2] - current_data.x[:, :2]  # Position difference
        
        # Compute loss
        loss = criterion(pred_vel_update, target_vel_update)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Validate the model.
        
        Args:
            dataloader: Validation data loader.
            criterion: Loss function.
            
        Returns:
            Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                if isinstance(batch, list):
                    # Handle sequence data
                    for i in range(len(batch) - 1):
                        current_data = batch[i].to(self.device)
                        next_data = batch[i + 1].to(self.device)
                        
                        loss = self._val_step(current_data, next_data, criterion)
                        total_loss += loss
                        num_batches += 1
                else:
                    # Handle single graph data
                    batch = batch.to(self.device)
                    loss = self._val_step(batch, batch, criterion)
                    total_loss += loss
                    num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _val_step(
        self,
        current_data,
        next_data,
        criterion: nn.Module
    ) -> float:
        """Single validation step.
        
        Args:
            current_data: Current time step data.
            next_data: Next time step data.
            criterion: Loss function.
            
        Returns:
            Loss value.
        """
        # Forward pass
        pred_vel_update = self.model(
            current_data.x,
            current_data.edge_index,
            current_data.pos
        )
        
        # Compute target velocity update
        target_vel_update = next_data.x[:, :2] - current_data.x[:, :2]
        
        # Compute loss
        loss = criterion(pred_vel_update, target_vel_update)
        
        return loss.item()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        criterion: Optional[nn.Module] = None,
        save_best: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of training epochs.
            criterion: Loss function.
            save_best: Whether to save the best model.
            
        Returns:
            Dictionary containing training history.
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, criterion)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate(val_loader, criterion)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_model(
                    self.model,
                    os.path.join(self.checkpoint_dir, "best_model.pt"),
                    self.optimizer,
                    epoch,
                    val_loss
                )
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Checkpoint metadata.
        """
        return load_model(self.model, checkpoint_path, self.optimizer, self.device)


def create_data_loaders(
    batch_size: int = 32,
    num_samples: int = 1000,
    num_particles: int = 50,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        batch_size: Batch size for data loaders.
        num_samples: Number of simulation samples.
        num_particles: Number of particles per simulation.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Create synthetic data
    train_data, test_data = create_synthetic_physics_data(
        num_samples=num_samples,
        num_particles=num_particles
    )
    
    # Split training data into train/val
    val_size = int(len(train_data) * val_split)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_physics_model(
    model_type: str = "egnn",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    num_layers: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[PhysicsGNN, Dict[str, List[float]]]:
    """Train a physics simulation model.
    
    Args:
        model_type: Type of model ("gcn", "gat", "egnn").
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        hidden_dim: Hidden dimension.
        num_layers: Number of layers.
        device: Device to train on.
        
    Returns:
        Tuple of (trained_model, training_history).
    """
    # Create model
    model = PhysicsGNN(
        input_dim=5,  # pos(2) + vel(2) + mass(1)
        hidden_dim=hidden_dim,
        output_dim=2,  # velocity update
        num_layers=num_layers,
        model_type=model_type
    )
    
    # Create trainer
    trainer = PhysicsTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=batch_size
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )
    
    return model, history
