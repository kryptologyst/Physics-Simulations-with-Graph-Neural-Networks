#!/usr/bin/env python3
"""Modernized physics simulation with GNNs - Main script."""

import argparse
import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.physics_gnn import PhysicsGNN
from data.physics_data import ParticleSystem
from train.trainer import PhysicsTrainer, create_data_loaders
from eval.evaluator import PhysicsEvaluator, create_evaluation_report
from utils.device import get_device, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Physics Simulation with GNNs")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gcn", "gat", "egnn"],
        default="egnn",
        help="Type of GNN model to use"
    )
    parser.add_argument(
        "--num-particles",
        type=int,
        default=50,
        help="Number of particles in simulation"
    )
    parser.add_argument(
        "--interaction-type",
        type=str,
        choices=["gravitational", "spring", "repulsive"],
        default="gravitational",
        help="Type of particle interaction"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of simulation steps"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model before simulation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save visualization plot"
    )
    
    return parser.parse_args()


def train_model(model_type: str, epochs: int, seed: int) -> PhysicsGNN:
    """Train a physics simulation model.
    
    Args:
        model_type: Type of model to train.
        epochs: Number of training epochs.
        seed: Random seed.
        
    Returns:
        Trained model.
    """
    print(f"Training {model_type.upper()} model...")
    
    # Create model
    model = PhysicsGNN(
        input_dim=5,  # pos(2) + vel(2) + mass(1)
        hidden_dim=64,
        output_dim=2,  # velocity update
        num_layers=3,
        model_type=model_type
    )
    
    # Create trainer
    trainer = PhysicsTrainer(
        model=model,
        learning_rate=0.001
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=32,
        num_samples=500,
        num_particles=50
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs
    )
    
    print(f"Training completed! Best validation loss: {trainer.best_val_loss:.4f}")
    
    return model


def run_simulation(
    model: PhysicsGNN,
    num_particles: int,
    interaction_type: str,
    num_steps: int,
    seed: int
) -> tuple:
    """Run physics simulation with the trained model.
    
    Args:
        model: Trained model.
        num_particles: Number of particles.
        interaction_type: Type of interaction.
        num_steps: Number of simulation steps.
        seed: Random seed.
        
    Returns:
        Tuple of (positions_history, velocities_history, energies_history).
    """
    print(f"Running simulation with {num_particles} particles...")
    
    # Create particle system
    system = ParticleSystem(
        num_particles=num_particles,
        dim=2,
        interaction_type=interaction_type,
        seed=seed
    )
    
    # Store simulation data
    positions_history = []
    velocities_history = []
    energies_history = []
    
    model.eval()
    device = get_device()
    
    with torch.no_grad():
        for step in range(num_steps):
            # Store current state
            positions_history.append(system.positions.clone())
            velocities_history.append(system.velocities.clone())
            
            # Compute energy
            kinetic_energy = 0.5 * torch.sum(
                system.masses.unsqueeze(-1) * system.velocities**2
            )
            energies_history.append(kinetic_energy.item())
            
            # Advance simulation
            system.step()
    
    return positions_history, velocities_history, energies_history


def visualize_results(
    positions_history: list,
    velocities_history: list,
    energies_history: list,
    model_type: str,
    save_plot: bool = False
) -> None:
    """Visualize simulation results.
    
    Args:
        positions_history: List of position tensors.
        velocities_history: List of velocity tensors.
        energies_history: List of energy values.
        model_type: Type of model used.
        save_plot: Whether to save the plot.
    """
    print("Creating visualization...")
    
    # Convert to numpy arrays
    positions_array = torch.stack(positions_history).numpy()
    velocities_array = torch.stack(velocities_history).numpy()
    energies_array = np.array(energies_history)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Physics Simulation Results - {model_type.upper()}", fontsize=16)
    
    # Plot 1: Particle trajectories
    ax1 = axes[0, 0]
    for i in range(min(positions_array.shape[1], 20)):  # Show first 20 particles
        ax1.plot(positions_array[:, i, 0], positions_array[:, i, 1], 
                alpha=0.7, linewidth=1)
    
    # Mark start and end positions
    ax1.scatter(positions_array[0, :, 0], positions_array[0, :, 1], 
               c='green', s=50, label='Start', alpha=0.8)
    ax1.scatter(positions_array[-1, :, 0], positions_array[-1, :, 1], 
               c='red', s=50, label='End', alpha=0.8)
    
    ax1.set_title("Particle Trajectories")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Velocity field
    ax2 = axes[0, 1]
    final_positions = positions_array[-1]
    final_velocities = velocities_array[-1]
    
    # Create velocity field plot
    ax2.quiver(final_positions[:, 0], final_positions[:, 1],
               final_velocities[:, 0], final_velocities[:, 1],
               alpha=0.7, scale=20)
    
    ax2.set_title("Final Velocity Field")
    ax2.set_xlabel("X Position")
    ax2.set_ylabel("Y Position")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Energy evolution
    ax3 = axes[1, 0]
    ax3.plot(energies_array, 'b-', linewidth=2)
    ax3.set_title("Energy Evolution")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Kinetic Energy")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Position distribution
    ax4 = axes[1, 1]
    final_positions = positions_array[-1]
    ax4.hist2d(final_positions[:, 0], final_positions[:, 1], 
               bins=20, alpha=0.7, cmap='viridis')
    ax4.set_title("Final Position Distribution")
    ax4.set_xlabel("X Position")
    ax4.set_ylabel("Y Position")
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("assets", exist_ok=True)
        plt.savefig(f"assets/physics_simulation_{model_type}.png", 
                   dpi=300, bbox_inches='tight')
        print(f"Plot saved to assets/physics_simulation_{model_type}.png")
    
    plt.show()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    print("Physics Simulation with Graph Neural Networks")
    print("=" * 50)
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Particles: {args.num_particles}")
    print(f"Interaction: {args.interaction_type}")
    print(f"Steps: {args.num_steps}")
    print(f"Seed: {args.seed}")
    print()
    
    # Train model if requested
    if args.train:
        model = train_model(args.model_type, args.epochs, args.seed)
    else:
        # Create untrained model for demonstration
        model = PhysicsGNN(
            input_dim=5,
            hidden_dim=64,
            output_dim=2,
            num_layers=3,
            model_type=args.model_type
        )
        print(f"Using untrained {args.model_type.upper()} model")
    
    # Run simulation
    positions_history, velocities_history, energies_history = run_simulation(
        model=model,
        num_particles=args.num_particles,
        interaction_type=args.interaction_type,
        num_steps=args.num_steps,
        seed=args.seed
    )
    
    # Visualize results
    visualize_results(
        positions_history=positions_history,
        velocities_history=velocities_history,
        energies_history=energies_history,
        model_type=args.model_type,
        save_plot=args.save_plot
    )
    
    # Print statistics
    print("\nSimulation Statistics:")
    print(f"Final Kinetic Energy: {energies_history[-1]:.2f}")
    print(f"Energy Conservation (std): {np.std(energies_history):.2f}")
    print(f"Max Velocity: {np.max(np.linalg.norm(velocities_history[-1], axis=1)):.2f}")
    print(f"Position Spread: {np.std(positions_history[-1]):.2f}")


if __name__ == "__main__":
    main()
