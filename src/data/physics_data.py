"""Data generation and processing utilities for physics simulations."""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class ParticleSystem:
    """A particle system for physics simulations.
    
    This class manages particles in 2D/3D space with various interaction types
    including gravitational, spring, and repulsive forces.
    """
    
    def __init__(
        self,
        num_particles: int = 50,
        dim: int = 2,
        interaction_type: str = "gravitational",
        dt: float = 0.01,
        gravity: float = 9.81,
        spring_constant: float = 1.0,
        damping: float = 0.99,
        boundary_size: float = 10.0,
        seed: Optional[int] = None
    ):
        """Initialize particle system.
        
        Args:
            num_particles: Number of particles in the system.
            dim: Dimension of the space (2 or 3).
            interaction_type: Type of interaction ("gravitational", "spring", "repulsive").
            dt: Time step for simulation.
            gravity: Gravitational constant.
            spring_constant: Spring constant for spring interactions.
            damping: Damping factor for velocity.
            boundary_size: Size of the simulation boundary.
            seed: Random seed for reproducibility.
        """
        self.num_particles = num_particles
        self.dim = dim
        self.interaction_type = interaction_type
        self.dt = dt
        self.gravity = gravity
        self.spring_constant = spring_constant
        self.damping = damping
        self.boundary_size = boundary_size
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Initialize particle states
        self.positions = torch.rand(num_particles, dim) * boundary_size
        self.velocities = torch.zeros(num_particles, dim)
        self.masses = torch.ones(num_particles) * 0.1
        
        # Create fully connected graph
        self.edge_index = self._create_edge_index()
        
    def _create_edge_index(self) -> torch.Tensor:
        """Create edge index for fully connected graph."""
        edge_index = torch.combinations(torch.arange(self.num_particles), r=2).t()
        # Make bidirectional
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index
    
    def compute_forces(self) -> torch.Tensor:
        """Compute forces between particles based on interaction type.
        
        Returns:
            torch.Tensor: Force vectors for each particle.
        """
        forces = torch.zeros(self.num_particles, self.dim)
        
        for i in range(self.num_particles):
            for j in range(self.num_particles):
                if i != j:
                    # Distance vector
                    r_vec = self.positions[j] - self.positions[i]
                    r = torch.norm(r_vec)
                    
                    if r > 0:  # Avoid division by zero
                        r_hat = r_vec / r
                        
                        if self.interaction_type == "gravitational":
                            # Gravitational force
                            force_magnitude = self.gravity * self.masses[i] * self.masses[j] / (r**2 + 1e-6)
                            forces[i] += force_magnitude * r_hat
                            
                        elif self.interaction_type == "spring":
                            # Spring force
                            force_magnitude = self.spring_constant * r
                            forces[i] += force_magnitude * r_hat
                            
                        elif self.interaction_type == "repulsive":
                            # Repulsive force
                            force_magnitude = self.spring_constant / (r**2 + 1e-6)
                            forces[i] -= force_magnitude * r_hat
                            
        return forces
    
    def step(self) -> None:
        """Advance the simulation by one time step."""
        forces = self.compute_forces()
        
        # Update velocities
        self.velocities += forces * self.dt / self.masses.unsqueeze(1)
        self.velocities *= self.damping
        
        # Update positions
        self.positions += self.velocities * self.dt
        
        # Apply boundary conditions (bounce)
        for d in range(self.dim):
            mask = self.positions[:, d] < 0
            self.positions[mask, d] = -self.positions[mask, d]
            self.velocities[mask, d] = -self.velocities[mask, d]
            
            mask = self.positions[:, d] > self.boundary_size
            self.positions[mask, d] = 2 * self.boundary_size - self.positions[mask, d]
            self.velocities[mask, d] = -self.velocities[mask, d]
    
    def to_graph_data(self) -> Data:
        """Convert particle system to PyTorch Geometric Data object.
        
        Returns:
            Data: Graph representation of the particle system.
        """
        # Node features: [position, velocity, mass]
        node_features = torch.cat([
            self.positions,
            self.velocities,
            self.masses.unsqueeze(1)
        ], dim=1)
        
        return Data(
            x=node_features,
            edge_index=self.edge_index,
            pos=self.positions
        )


class PhysicsDataset(Dataset):
    """Dataset for physics simulation data."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_particles: int = 50,
        dim: int = 2,
        interaction_type: str = "gravitational",
        sequence_length: int = 10,
        root: Optional[str] = None,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
        seed: Optional[int] = None
    ):
        """Initialize physics dataset.
        
        Args:
            num_samples: Number of simulation sequences to generate.
            num_particles: Number of particles per simulation.
            dim: Dimension of the space.
            interaction_type: Type of particle interaction.
            sequence_length: Length of each simulation sequence.
            root: Root directory for the dataset.
            transform: Transform to apply to each sample.
            pre_transform: Pre-transform to apply to each sample.
            seed: Random seed for reproducibility.
        """
        self.num_samples = num_samples
        self.num_particles = num_particles
        self.dim = dim
        self.interaction_type = interaction_type
        self.sequence_length = sequence_length
        self.seed = seed
        
        super().__init__(root, transform, pre_transform)
        
        # Generate data if not already cached
        if not self._is_cached():
            self._generate_data()
    
    def _is_cached(self) -> bool:
        """Check if data is already cached."""
        return len(self.processed_file_names) > 0
    
    def _generate_data(self) -> None:
        """Generate physics simulation data."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            
        for i in range(self.num_samples):
            # Create particle system
            system = ParticleSystem(
                num_particles=self.num_particles,
                dim=self.dim,
                interaction_type=self.interaction_type,
                seed=self.seed
            )
            
            # Generate sequence
            sequence = []
            for _ in range(self.sequence_length):
                data = system.to_graph_data()
                sequence.append(data)
                system.step()
            
            # Save sequence
            torch.save(sequence, self.processed_paths[i])
    
    @property
    def processed_file_names(self) -> List[str]:
        """Get list of processed file names."""
        return [f"sequence_{i}.pt" for i in range(self.num_samples)]
    
    def __len__(self) -> int:
        """Get dataset length."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> List[Data]:
        """Get a sequence from the dataset."""
        sequence = torch.load(self.processed_paths[idx])
        if self.transform:
            sequence = [self.transform(data) for data in sequence]
        return sequence


def create_synthetic_physics_data(
    num_samples: int = 100,
    num_particles: int = 20,
    dim: int = 2,
    interaction_type: str = "gravitational"
) -> Tuple[List[Data], List[Data]]:
    """Create synthetic physics simulation data for training and testing.
    
    Args:
        num_samples: Number of simulation sequences.
        num_particles: Number of particles per simulation.
        dim: Dimension of the space.
        interaction_type: Type of particle interaction.
        
    Returns:
        Tuple of (train_data, test_data) lists.
    """
    train_data = []
    test_data = []
    
    for i in range(num_samples):
        system = ParticleSystem(
            num_particles=num_particles,
            dim=dim,
            interaction_type=interaction_type,
            seed=i
        )
        
        # Generate sequence
        sequence = []
        for _ in range(20):  # 20 time steps
            data = system.to_graph_data()
            sequence.append(data)
            system.step()
        
        # Split into train/test
        if i < num_samples * 0.8:
            train_data.extend(sequence)
        else:
            test_data.extend(sequence)
    
    return train_data, test_data
