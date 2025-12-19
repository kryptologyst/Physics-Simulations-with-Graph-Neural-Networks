"""Test suite for physics simulation models."""

import pytest
import torch
import numpy as np

from src.models.physics_gnn import PhysicsGNN, EGNNLayer, SE3TransformerLayer
from src.data.physics_data import ParticleSystem, PhysicsDataset
from src.utils.device import get_device, set_seed


class TestPhysicsGNN:
    """Test cases for PhysicsGNN model."""
    
    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        # Test GCN model
        model_gcn = PhysicsGNN(
            input_dim=5,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            model_type="gcn"
        )
        assert model_gcn.model_type == "gcn"
        assert model_gcn.input_dim == 5
        assert model_gcn.output_dim == 2
        
        # Test GAT model
        model_gat = PhysicsGNN(
            input_dim=5,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            model_type="gat"
        )
        assert model_gat.model_type == "gat"
        
        # Test EGNN model
        model_egnn = PhysicsGNN(
            input_dim=5,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            model_type="egnn"
        )
        assert model_egnn.model_type == "egnn"
    
    def test_forward_pass(self):
        """Test forward pass with different model types."""
        batch_size = 10
        num_nodes = 20
        input_dim = 5
        output_dim = 2
        
        # Create dummy data
        x = torch.randn(num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, 50))
        pos = torch.randn(num_nodes, 2)
        
        for model_type in ["gcn", "gat", "egnn"]:
            model = PhysicsGNN(
                input_dim=input_dim,
                hidden_dim=32,
                output_dim=output_dim,
                num_layers=2,
                model_type=model_type
            )
            
            # Forward pass
            if model_type == "egnn":
                output = model(x, edge_index, pos)
            else:
                output = model(x, edge_index)
            
            assert output.shape == (num_nodes, output_dim)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_energy_computation(self):
        """Test energy computation functionality."""
        model = PhysicsGNN(model_type="egnn")
        
        # Create test data
        pos = torch.randn(10, 2)
        vel = torch.randn(10, 2)
        masses = torch.ones(10, 1)
        
        energy = model.compute_energy(pos, vel, masses)
        
        assert isinstance(energy, torch.Tensor)
        assert energy.item() >= 0  # Energy should be non-negative
        assert not torch.isnan(energy)
        assert not torch.isinf(energy)


class TestEGNNLayer:
    """Test cases for EGNN layer."""
    
    def test_layer_initialization(self):
        """Test EGNN layer initialization."""
        layer = EGNNLayer(
            in_channels=32,
            hidden_channels=64,
            out_channels=32
        )
        
        assert layer.in_channels == 32
        assert layer.hidden_channels == 64
        assert layer.out_channels == 32
    
    def test_forward_pass(self):
        """Test EGNN layer forward pass."""
        layer = EGNNLayer(
            in_channels=32,
            hidden_channels=64,
            out_channels=32
        )
        
        # Create test data
        x = torch.randn(20, 32)
        pos = torch.randn(20, 2)
        edge_index = torch.randint(0, 20, (2, 50))
        
        # Forward pass
        x_new, pos_new = layer(x, pos, edge_index)
        
        assert x_new.shape == (20, 32)
        assert pos_new.shape == (20, 2)
        assert not torch.isnan(x_new).any()
        assert not torch.isnan(pos_new).any()


class TestParticleSystem:
    """Test cases for ParticleSystem."""
    
    def test_system_initialization(self):
        """Test particle system initialization."""
        system = ParticleSystem(
            num_particles=20,
            dim=2,
            interaction_type="gravitational"
        )
        
        assert system.num_particles == 20
        assert system.dim == 2
        assert system.interaction_type == "gravitational"
        assert system.positions.shape == (20, 2)
        assert system.velocities.shape == (20, 2)
        assert system.masses.shape == (20,)
    
    def test_edge_index_creation(self):
        """Test edge index creation."""
        system = ParticleSystem(num_particles=5)
        
        # Should create fully connected graph
        expected_edges = 5 * 4  # 5 nodes, 4 edges per node (bidirectional)
        assert system.edge_index.shape == (2, expected_edges)
        
        # Check that all nodes are connected
        unique_nodes = torch.unique(system.edge_index)
        assert len(unique_nodes) == 5
    
    def test_force_computation(self):
        """Test force computation."""
        system = ParticleSystem(
            num_particles=5,
            interaction_type="gravitational"
        )
        
        forces = system.compute_forces()
        
        assert forces.shape == (5, 2)
        assert not torch.isnan(forces).any()
        assert not torch.isinf(forces).any()
    
    def test_simulation_step(self):
        """Test simulation step."""
        system = ParticleSystem(num_particles=5)
        
        initial_pos = system.positions.clone()
        initial_vel = system.velocities.clone()
        
        system.step()
        
        # Positions and velocities should change
        assert not torch.allclose(system.positions, initial_pos)
        assert not torch.allclose(system.velocities, initial_vel)
    
    def test_to_graph_data(self):
        """Test conversion to PyTorch Geometric Data."""
        system = ParticleSystem(num_particles=5)
        
        data = system.to_graph_data()
        
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'pos')
        
        # Node features should include pos(2) + vel(2) + mass(1) = 5
        assert data.x.shape == (5, 5)
        assert data.pos.shape == (5, 2)
        assert data.edge_index.shape[0] == 2


class TestPhysicsDataset:
    """Test cases for PhysicsDataset."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = PhysicsDataset(
            num_samples=10,
            num_particles=5,
            sequence_length=5,
            seed=42
        )
        
        assert len(dataset) == 10
        assert dataset.num_samples == 10
        assert dataset.num_particles == 5
        assert dataset.sequence_length == 5
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        dataset = PhysicsDataset(
            num_samples=5,
            num_particles=3,
            sequence_length=3,
            seed=42
        )
        
        sequence = dataset[0]
        
        assert len(sequence) == 3
        assert all(hasattr(data, 'x') for data in sequence)
        assert all(hasattr(data, 'edge_index') for data in sequence)
        assert all(hasattr(data, 'pos') for data in sequence)


class TestDeviceUtils:
    """Test cases for device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seed is set
        torch_rand = torch.rand(1)
        np_rand = np.random.rand(1)
        
        # Reset seed and test reproducibility
        set_seed(42)
        torch_rand2 = torch.rand(1)
        np_rand2 = np.random.rand(1)
        
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training process."""
        # Create small dataset
        dataset = PhysicsDataset(
            num_samples=5,
            num_particles=3,
            sequence_length=3,
            seed=42
        )
        
        # Create model
        model = PhysicsGNN(
            input_dim=5,
            hidden_dim=16,
            output_dim=2,
            num_layers=2,
            model_type="gcn"
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(3):
            total_loss = 0
            for i in range(len(dataset)):
                sequence = dataset[i]
                
                for j in range(len(sequence) - 1):
                    current_data = sequence[j]
                    next_data = sequence[j + 1]
                    
                    optimizer.zero_grad()
                    
                    pred = model(current_data.x, current_data.edge_index)
                    target = next_data.x[:, :2] - current_data.x[:, :2]
                    
                    loss = criterion(pred, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            
            assert total_loss >= 0  # Loss should be non-negative
    
    def test_model_comparison(self):
        """Test comparison between different model types."""
        # Create test data
        x = torch.randn(10, 5)
        edge_index = torch.randint(0, 10, (2, 20))
        pos = torch.randn(10, 2)
        
        models = {}
        outputs = {}
        
        for model_type in ["gcn", "gat", "egnn"]:
            model = PhysicsGNN(
                input_dim=5,
                hidden_dim=16,
                output_dim=2,
                num_layers=2,
                model_type=model_type
            )
            
            if model_type == "egnn":
                output = model(x, edge_index, pos)
            else:
                output = model(x, edge_index)
            
            models[model_type] = model
            outputs[model_type] = output
            
            assert output.shape == (10, 2)
            assert not torch.isnan(output).any()
        
        # All models should produce different outputs
        gcn_output = outputs["gcn"]
        gat_output = outputs["gat"]
        egnn_output = outputs["egnn"]
        
        assert not torch.allclose(gcn_output, gat_output)
        assert not torch.allclose(gcn_output, egnn_output)
        assert not torch.allclose(gat_output, egnn_output)


if __name__ == "__main__":
    pytest.main([__file__])
