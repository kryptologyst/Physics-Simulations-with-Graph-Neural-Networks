#!/usr/bin/env python3
"""Main training script for physics simulation models."""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.physics_gnn import PhysicsGNN
from train.trainer import PhysicsTrainer, create_data_loaders, train_physics_model
from eval.evaluator import PhysicsEvaluator, create_evaluation_report
from utils.device import get_device, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train physics simulation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gcn", "gat", "egnn"],
        help="Override model type from config"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs from config"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate from config"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Override device from config"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed from config"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate existing model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint for evaluation"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_type:
        config["model"]["type"] = args.model_type
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.device:
        config["device"]["auto_detect"] = False
        config["device"]["manual"] = args.device
    if args.seed:
        config["seed"] = args.seed
    
    # Set random seed
    set_seed(config["seed"])
    
    # Get device
    if config["device"].get("auto_detect", True):
        device = get_device()
    else:
        device = torch.device(config["device"]["manual"])
    
    print(f"Using device: {device}")
    
    # Create model
    model_config = config["model"]
    model = PhysicsGNN(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        output_dim=model_config["output_dim"],
        num_layers=model_config["num_layers"],
        model_type=model_config["type"],
        dropout=model_config["dropout"],
        use_residual=model_config["use_residual"]
    )
    
    print(f"Model: {model_config['type'].upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if args.eval_only:
        # Evaluation only mode
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation mode")
            sys.exit(1)
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Create test data
        _, _, test_loader = create_data_loaders(
            batch_size=config["training"]["batch_size"],
            num_samples=config["data"]["num_samples"],
            num_particles=config["data"]["num_particles"]
        )
        
        # Evaluate model
        evaluator = PhysicsEvaluator(model, device)
        test_data = list(test_loader.dataset)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        report = create_evaluation_report(model, test_data, device)
        print(report)
        
    else:
        # Training mode
        print("\nStarting training...")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=config["training"]["batch_size"],
            num_samples=config["data"]["num_samples"],
            num_particles=config["data"]["num_particles"],
            train_split=config["data"]["train_split"],
            val_split=config["data"]["val_split"]
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Train model
        trainer = PhysicsTrainer(
            model=model,
            device=device,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            checkpoint_dir=config["checkpoints"]["save_dir"]
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["training"]["num_epochs"]
        )
        
        print("\nTraining completed!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_data = list(test_loader.dataset)
        
        evaluator = PhysicsEvaluator(model, device)
        test_metrics = evaluator.comprehensive_evaluation(test_data)
        
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        
        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Create evaluation report
        report = create_evaluation_report(model, test_data, device)
        
        # Save report
        os.makedirs("assets", exist_ok=True)
        with open("assets/evaluation_report.txt", "w") as f:
            f.write(report)
        
        print(f"\nEvaluation report saved to assets/evaluation_report.txt")


if __name__ == "__main__":
    main()
