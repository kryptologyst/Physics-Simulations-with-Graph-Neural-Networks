"""Evaluation metrics and utilities for physics simulations."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from ..models.physics_gnn import PhysicsGNN
from ..utils.device import get_device


class PhysicsEvaluator:
    """Evaluator for physics simulation models."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate.
            device: Device to evaluate on.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize metrics
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        
    def evaluate_rollout(
        self,
        initial_data,
        num_steps: int = 50,
        dt: float = 0.01
    ) -> Dict[str, float]:
        """Evaluate model on rollout prediction.
        
        Args:
            initial_data: Initial state of the system.
            num_steps: Number of rollout steps.
            dt: Time step size.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        
        # Initialize rollout
        current_data = initial_data.to(self.device)
        rollout_positions = [current_data.pos.clone()]
        rollout_velocities = [current_data.x[:, :2].clone()]
        
        total_error = 0.0
        energy_errors = []
        
        with torch.no_grad():
            for step in range(num_steps):
                # Predict velocity update
                vel_update = self.model(
                    current_data.x,
                    current_data.edge_index,
                    current_data.pos
                )
                
                # Update positions and velocities
                new_vel = current_data.x[:, :2] + vel_update * dt
                new_pos = current_data.pos + new_vel * dt
                
                # Update data
                current_data.x[:, :2] = new_vel
                current_data.pos = new_pos
                
                # Store rollout
                rollout_positions.append(new_pos.clone())
                rollout_velocities.append(new_vel.clone())
                
                # Compute energy error
                if hasattr(self.model, 'compute_energy'):
                    masses = current_data.x[:, 4:5]  # Assuming mass is 5th feature
                    energy = self.model.compute_energy(new_pos, new_vel, masses)
                    energy_errors.append(energy.item())
        
        # Compute metrics
        rollout_positions = torch.stack(rollout_positions)
        rollout_velocities = torch.stack(rollout_velocities)
        
        # Position stability (variance of final positions)
        final_positions = rollout_positions[-1]
        position_stability = torch.var(final_positions).item()
        
        # Velocity stability
        final_velocities = rollout_velocities[-1]
        velocity_stability = torch.var(final_velocities).item()
        
        # Energy conservation
        energy_conservation = np.std(energy_errors) if energy_errors else 0.0
        
        return {
            "position_stability": position_stability,
            "velocity_stability": velocity_stability,
            "energy_conservation": energy_conservation,
            "rollout_length": num_steps
        }
    
    def evaluate_physics_constraints(
        self,
        data,
        predictions: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate physics constraint violations.
        
        Args:
            data: Input data.
            predictions: Model predictions.
            
        Returns:
            Dictionary of constraint violation metrics.
        """
        # Extract positions and velocities
        positions = data.pos
        velocities = data.x[:, :2]
        masses = data.x[:, 4:5] if data.x.size(1) > 4 else torch.ones(positions.size(0), 1)
        
        # Compute predicted velocities
        pred_velocities = velocities + predictions
        
        # Momentum conservation
        initial_momentum = torch.sum(masses * velocities, dim=0)
        final_momentum = torch.sum(masses * pred_velocities, dim=0)
        momentum_error = torch.norm(final_momentum - initial_momentum).item()
        
        # Angular momentum conservation
        initial_angular_momentum = torch.sum(
            masses * torch.cross(positions, velocities, dim=1)
        ).item()
        final_angular_momentum = torch.sum(
            masses * torch.cross(positions, pred_velocities, dim=1)
        ).item()
        angular_momentum_error = abs(final_angular_momentum - initial_angular_momentum)
        
        # Energy conservation
        initial_kinetic = 0.5 * torch.sum(masses * velocities**2).item()
        final_kinetic = 0.5 * torch.sum(masses * pred_velocities**2).item()
        energy_error = abs(final_kinetic - initial_kinetic)
        
        return {
            "momentum_error": momentum_error,
            "angular_momentum_error": angular_momentum_error,
            "energy_error": energy_error
        }
    
    def evaluate_trajectory_accuracy(
        self,
        true_trajectory: List[torch.Tensor],
        pred_trajectory: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate trajectory prediction accuracy.
        
        Args:
            true_trajectory: Ground truth trajectory.
            pred_trajectory: Predicted trajectory.
            
        Returns:
            Dictionary of accuracy metrics.
        """
        if len(true_trajectory) != len(pred_trajectory):
            raise ValueError("Trajectory lengths must match")
        
        # Compute position errors
        position_errors = []
        velocity_errors = []
        
        for true_state, pred_state in zip(true_trajectory, pred_trajectory):
            if true_state.size(0) != pred_state.size(0):
                continue
                
            # Position error
            pos_error = torch.norm(true_state.pos - pred_state.pos, dim=1)
            position_errors.append(pos_error.mean().item())
            
            # Velocity error
            vel_error = torch.norm(true_state.x[:, :2] - pred_state.x[:, :2], dim=1)
            velocity_errors.append(vel_error.mean().item())
        
        return {
            "mean_position_error": np.mean(position_errors),
            "std_position_error": np.std(position_errors),
            "mean_velocity_error": np.mean(velocity_errors),
            "std_velocity_error": np.std(velocity_errors),
            "max_position_error": np.max(position_errors),
            "max_velocity_error": np.max(velocity_errors)
        }
    
    def evaluate_long_term_stability(
        self,
        initial_data,
        num_steps: int = 1000,
        dt: float = 0.01
    ) -> Dict[str, float]:
        """Evaluate long-term stability of the simulation.
        
        Args:
            initial_data: Initial state.
            num_steps: Number of simulation steps.
            dt: Time step size.
            
        Returns:
            Dictionary of stability metrics.
        """
        self.model.eval()
        
        current_data = initial_data.to(self.device)
        positions_history = []
        velocities_history = []
        energies_history = []
        
        with torch.no_grad():
            for step in range(num_steps):
                # Predict velocity update
                vel_update = self.model(
                    current_data.x,
                    current_data.edge_index,
                    current_data.pos
                )
                
                # Update state
                new_vel = current_data.x[:, :2] + vel_update * dt
                new_pos = current_data.pos + new_vel * dt
                
                current_data.x[:, :2] = new_vel
                current_data.pos = new_pos
                
                # Store history
                positions_history.append(new_pos.clone())
                velocities_history.append(new_vel.clone())
                
                # Compute energy
                if hasattr(self.model, 'compute_energy'):
                    masses = current_data.x[:, 4:5]
                    energy = self.model.compute_energy(new_pos, new_vel, masses)
                    energies_history.append(energy.item())
        
        # Analyze stability
        positions_tensor = torch.stack(positions_history)
        velocities_tensor = torch.stack(velocities_history)
        
        # Position drift
        initial_pos = positions_tensor[0]
        final_pos = positions_tensor[-1]
        position_drift = torch.norm(final_pos - initial_pos, dim=1).mean().item()
        
        # Velocity drift
        initial_vel = velocities_tensor[0]
        final_vel = velocities_tensor[-1]
        velocity_drift = torch.norm(final_vel - initial_vel, dim=1).mean().item()
        
        # Energy drift
        if energies_history:
            energy_drift = abs(energies_history[-1] - energies_history[0])
        else:
            energy_drift = 0.0
        
        # Oscillation analysis
        position_variance = torch.var(positions_tensor, dim=0).mean().item()
        velocity_variance = torch.var(velocities_tensor, dim=0).mean().item()
        
        return {
            "position_drift": position_drift,
            "velocity_drift": velocity_drift,
            "energy_drift": energy_drift,
            "position_variance": position_variance,
            "velocity_variance": velocity_variance,
            "simulation_steps": num_steps
        }
    
    def comprehensive_evaluation(
        self,
        test_data: List,
        num_rollout_steps: int = 50,
        num_stability_steps: int = 1000
    ) -> Dict[str, float]:
        """Perform comprehensive evaluation of the model.
        
        Args:
            test_data: Test dataset.
            num_rollout_steps: Number of steps for rollout evaluation.
            num_stability_steps: Number of steps for stability evaluation.
            
        Returns:
            Dictionary of comprehensive evaluation metrics.
        """
        all_metrics = {}
        
        # Rollout evaluation
        rollout_metrics = []
        for data in test_data[:10]:  # Evaluate on first 10 samples
            metrics = self.evaluate_rollout(data, num_rollout_steps)
            rollout_metrics.append(metrics)
        
        # Average rollout metrics
        for key in rollout_metrics[0].keys():
            all_metrics[f"rollout_{key}"] = np.mean([m[key] for m in rollout_metrics])
        
        # Stability evaluation
        stability_metrics = []
        for data in test_data[:5]:  # Evaluate on first 5 samples
            metrics = self.evaluate_long_term_stability(data, num_stability_steps)
            stability_metrics.append(metrics)
        
        # Average stability metrics
        for key in stability_metrics[0].keys():
            all_metrics[f"stability_{key}"] = np.mean([m[key] for m in stability_metrics])
        
        return all_metrics


def create_evaluation_report(
    model: PhysicsGNN,
    test_data: List,
    device: Optional[torch.device] = None
) -> str:
    """Create a comprehensive evaluation report.
    
    Args:
        model: Trained model.
        test_data: Test dataset.
        device: Device to evaluate on.
        
    Returns:
        Formatted evaluation report.
    """
    evaluator = PhysicsEvaluator(model, device)
    
    # Comprehensive evaluation
    metrics = evaluator.comprehensive_evaluation(test_data)
    
    # Create report
    report = "Physics Simulation Model Evaluation Report\n"
    report += "=" * 50 + "\n\n"
    
    report += "Rollout Performance:\n"
    report += f"  Position Stability: {metrics['rollout_position_stability']:.4f}\n"
    report += f"  Velocity Stability: {metrics['rollout_velocity_stability']:.4f}\n"
    report += f"  Energy Conservation: {metrics['rollout_energy_conservation']:.4f}\n\n"
    
    report += "Long-term Stability:\n"
    report += f"  Position Drift: {metrics['stability_position_drift']:.4f}\n"
    report += f"  Velocity Drift: {metrics['stability_velocity_drift']:.4f}\n"
    report += f"  Energy Drift: {metrics['stability_energy_drift']:.4f}\n"
    report += f"  Position Variance: {metrics['stability_position_variance']:.4f}\n"
    report += f"  Velocity Variance: {metrics['stability_velocity_variance']:.4f}\n\n"
    
    report += "Model Parameters:\n"
    report += f"  Total Parameters: {sum(p.numel() for p in model.parameters())}\n"
    report += f"  Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n"
    
    return report
