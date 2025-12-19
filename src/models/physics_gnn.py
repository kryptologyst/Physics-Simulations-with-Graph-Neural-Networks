"""Physics-aware Graph Neural Network models for simulation."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.utils import softmax


class EGNNLayer(MessagePassing):
    """E(n) Equivariant Graph Neural Network layer.
    
    This layer is equivariant to rotations and translations in n-dimensional space,
    making it ideal for physics simulations where the laws should be invariant
    to coordinate system transformations.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_attr_dim: int = 0,
        aggr: str = "add",
        **kwargs
    ):
        """Initialize EGNN layer.
        
        Args:
            in_channels: Number of input features.
            hidden_channels: Number of hidden features.
            out_channels: Number of output features.
            edge_attr_dim: Dimension of edge attributes.
            aggr: Aggregation method.
        """
        super().__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_attr_dim = edge_attr_dim
        
        # Node feature transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # Edge feature transformation
        if edge_attr_dim > 0:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_attr_dim + 1, hidden_channels),  # +1 for distance
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )
        else:
            self.edge_mlp = nn.Sequential(
                nn.Linear(1, hidden_channels),  # Only distance
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )
        
        # Coordinate transformation
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.node_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.coord_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            pos: Node positions.
            edge_index: Edge indices.
            edge_attr: Edge attributes.
            
        Returns:
            Tuple of (updated_features, updated_positions).
        """
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute messages between nodes.
        
        Args:
            x_i: Source node features.
            x_j: Target node features.
            pos_i: Source node positions.
            pos_j: Target node positions.
            edge_attr: Edge attributes.
            
        Returns:
            Message tensor.
        """
        # Compute distance
        dist = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
        
        # Prepare edge features
        if edge_attr is not None:
            edge_input = torch.cat([edge_attr, dist], dim=-1)
        else:
            edge_input = dist
        
        # Compute edge weights
        edge_weight = self.edge_mlp(edge_input)
        
        # Compute coordinate updates
        coord_diff = pos_i - pos_j
        coord_input = torch.cat([x_i, x_j], dim=-1)
        coord_update = self.coord_mlp(coord_input) * coord_diff
        
        return edge_weight * coord_update
    
    def update(self, aggr_out: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update node features and positions.
        
        Args:
            aggr_out: Aggregated messages.
            pos: Current positions.
            
        Returns:
            Tuple of (updated_features, updated_positions).
        """
        # Update features
        x_new = self.node_mlp(aggr_out)
        
        # Update positions
        pos_new = pos + aggr_out
        
        return x_new, pos_new


class PhysicsGNN(nn.Module):
    """Physics-aware Graph Neural Network for particle dynamics simulation.
    
    This model combines traditional GNN layers with physics-aware components
    to predict particle dynamics while respecting physical constraints.
    """
    
    def __init__(
        self,
        input_dim: int = 5,  # pos(2) + vel(2) + mass(1)
        hidden_dim: int = 64,
        output_dim: int = 2,  # velocity update
        num_layers: int = 3,
        model_type: str = "egnn",
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """Initialize PhysicsGNN.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            output_dim: Output dimension (velocity update).
            num_layers: Number of GNN layers.
            model_type: Type of GNN ("gcn", "gat", "egnn").
            dropout: Dropout rate.
            use_residual: Whether to use residual connections.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.model_type = model_type
        self.dropout = dropout
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if model_type == "egnn":
                layer = EGNNLayer(
                    in_channels=hidden_dim,
                    hidden_channels=hidden_dim,
                    out_channels=hidden_dim
                )
            elif model_type == "gcn":
                layer = GCNConv(hidden_dim, hidden_dim)
            elif model_type == "gat":
                layer = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.gnn_layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Physics constraints
        self.velocity_constraint = nn.Parameter(torch.tensor(1.0))
        self.energy_constraint = nn.Parameter(torch.tensor(0.1))
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [batch_size, num_nodes, input_dim].
            edge_index: Edge indices [2, num_edges].
            pos: Node positions [batch_size, num_nodes, 2] (for EGNN).
            
        Returns:
            Velocity updates [batch_size, num_nodes, output_dim].
        """
        # Input projection
        h = self.input_proj(x)
        
        # GNN layers
        for i, layer in enumerate(self.gnn_layers):
            if self.model_type == "egnn" and pos is not None:
                h_new, pos_new = layer(h, pos, edge_index)
                h = h_new
                pos = pos_new
            else:
                h_new = layer(h, edge_index)
                h = h_new
            
            # Residual connection
            if self.use_residual and i > 0:
                h = h + h
            
            # Activation and dropout
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Output projection
        velocity_update = self.output_proj(h)
        
        # Apply physics constraints
        velocity_update = velocity_update * self.velocity_constraint
        
        return velocity_update
    
    def compute_energy(self, pos: torch.Tensor, vel: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Compute total energy of the system.
        
        Args:
            pos: Particle positions.
            vel: Particle velocities.
            masses: Particle masses.
            
        Returns:
            Total energy.
        """
        # Kinetic energy
        kinetic = 0.5 * torch.sum(masses.unsqueeze(-1) * vel**2)
        
        # Potential energy (simplified gravitational)
        potential = 0.0
        for i in range(pos.size(0)):
            for j in range(i + 1, pos.size(0)):
                dist = torch.norm(pos[i] - pos[j])
                potential += -masses[i] * masses[j] / (dist + 1e-6)
        
        return kinetic + potential


class SE3TransformerLayer(nn.Module):
    """SE(3) Transformer layer for 3D physics simulations.
    
    This layer is equivariant to rotations and translations in 3D space,
    making it suitable for 3D physics simulations.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """Initialize SE(3) Transformer layer.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden feature dimension.
            out_channels: Output feature dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(in_channels, hidden_channels)
        self.k_proj = nn.Linear(in_channels, hidden_channels)
        self.v_proj = nn.Linear(in_channels, hidden_channels)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        
        # Position encoding
        self.pos_encoding = nn.Linear(3, hidden_channels)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            pos: Node positions.
            edge_index: Edge indices.
            
        Returns:
            Updated node features.
        """
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Add position encoding
        pos_enc = self.pos_encoding(pos)
        q = q + pos_enc
        k = k + pos_enc
        
        # Compute attention weights
        attention_weights = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = attention_weights / math.sqrt(self.hidden_channels)
        
        # Apply attention
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, v)
        
        # Output projection
        out = self.out_proj(out)
        
        return out
