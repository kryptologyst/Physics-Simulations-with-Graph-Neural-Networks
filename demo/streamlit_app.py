"""Interactive Streamlit demo for physics simulation visualization."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.physics_gnn import PhysicsGNN
from data.physics_data import ParticleSystem
from eval.evaluator import PhysicsEvaluator
from utils.device import get_device, set_seed


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Physics Simulation with GNNs",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Physics Simulation with Graph Neural Networks")
    st.markdown("Interactive visualization and parameter tuning for physics simulation models")
    
    # Sidebar for parameters
    st.sidebar.header("Simulation Parameters")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["gcn", "gat", "egnn"],
        index=2,
        help="Type of Graph Neural Network model"
    )
    
    # Physics parameters
    st.sidebar.subheader("Physics Settings")
    num_particles = st.sidebar.slider("Number of Particles", 10, 100, 50)
    interaction_type = st.sidebar.selectbox(
        "Interaction Type",
        ["gravitational", "spring", "repulsive"],
        help="Type of particle interaction"
    )
    dt = st.sidebar.slider("Time Step", 0.001, 0.1, 0.01, 0.001)
    gravity = st.sidebar.slider("Gravity", 0.1, 20.0, 9.81, 0.1)
    damping = st.sidebar.slider("Damping", 0.8, 1.0, 0.99, 0.01)
    
    # Model parameters
    st.sidebar.subheader("Model Settings")
    hidden_dim = st.sidebar.slider("Hidden Dimension", 32, 128, 64)
    num_layers = st.sidebar.slider("Number of Layers", 1, 5, 3)
    dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
    
    # Simulation parameters
    st.sidebar.subheader("Simulation Settings")
    num_steps = st.sidebar.slider("Simulation Steps", 10, 200, 50)
    seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)
    
    # Set seed
    set_seed(seed)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Live Simulation", "Model Comparison", "Analysis", "About"])
    
    with tab1:
        st.header("Live Physics Simulation")
        
        # Create model
        model = PhysicsGNN(
            input_dim=5,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_layers=num_layers,
            model_type=model_type,
            dropout=dropout
        )
        
        # Create particle system
        system = ParticleSystem(
            num_particles=num_particles,
            dim=2,
            interaction_type=interaction_type,
            dt=dt,
            gravity=gravity,
            damping=damping,
            seed=seed
        )
        
        # Simulation controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Run Simulation", type="primary"):
                st.session_state.run_simulation = True
        with col2:
            if st.button("Reset"):
                st.session_state.run_simulation = False
                system = ParticleSystem(
                    num_particles=num_particles,
                    dim=2,
                    interaction_type=interaction_type,
                    dt=dt,
                    gravity=gravity,
                    damping=damping,
                    seed=seed
                )
        with col3:
            if st.button("Randomize"):
                st.session_state.run_simulation = False
                system = ParticleSystem(
                    num_particles=num_particles,
                    dim=2,
                    interaction_type=interaction_type,
                    dt=dt,
                    gravity=gravity,
                    damping=damping,
                    seed=None
                )
        
        # Run simulation
        if st.session_state.get("run_simulation", False):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Store simulation data
            positions_history = []
            velocities_history = []
            energies_history = []
            
            for step in range(num_steps):
                # Store current state
                positions_history.append(system.positions.clone())
                velocities_history.append(system.velocities.clone())
                
                # Compute energy
                kinetic_energy = 0.5 * torch.sum(system.masses.unsqueeze(-1) * system.velocities**2)
                energies_history.append(kinetic_energy.item())
                
                # Advance simulation
                system.step()
                
                # Update progress
                progress_bar.progress((step + 1) / num_steps)
                status_text.text(f"Step {step + 1}/{num_steps}")
            
            # Visualization
            st.subheader("Simulation Results")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Particle Trajectories", "Velocity Field", "Energy Evolution", "Particle Distribution"),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "histogram"}]]
            )
            
            # Particle trajectories
            positions_array = torch.stack(positions_history).numpy()
            for i in range(min(num_particles, 20)):  # Show first 20 particles
                fig.add_trace(
                    go.Scatter(
                        x=positions_array[:, i, 0],
                        y=positions_array[:, i, 1],
                        mode='lines',
                        name=f'Particle {i}',
                        line=dict(width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Final positions
            final_positions = positions_history[-1].numpy()
            fig.add_trace(
                go.Scatter(
                    x=final_positions[:, 0],
                    y=final_positions[:, 1],
                    mode='markers',
                    name='Final Positions',
                    marker=dict(size=8, color='red'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Velocity field
            final_velocities = velocities_history[-1].numpy()
            fig.add_trace(
                go.Scatter(
                    x=final_positions[:, 0],
                    y=final_positions[:, 1],
                    mode='markers+text',
                    text=[f'{np.linalg.norm(v):.2f}' for v in final_velocities],
                    textposition='top center',
                    marker=dict(size=10, color=np.linalg.norm(final_velocities, axis=1)),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Energy evolution
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(energies_history))),
                    y=energies_history,
                    mode='lines',
                    name='Kinetic Energy',
                    line=dict(color='blue'),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Particle distribution
            fig.add_trace(
                go.Histogram(
                    x=final_positions[:, 0],
                    name='X Distribution',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text="Physics Simulation Results",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Kinetic Energy", f"{energies_history[-1]:.2f}")
            with col2:
                st.metric("Max Velocity", f"{np.max(np.linalg.norm(final_velocities, axis=1)):.2f}")
            with col3:
                st.metric("Position Spread", f"{np.std(final_positions):.2f}")
            with col4:
                st.metric("Energy Conservation", f"{np.std(energies_history):.2f}")
    
    with tab2:
        st.header("Model Comparison")
        
        # Compare different models
        models_to_compare = st.multiselect(
            "Select models to compare",
            ["gcn", "gat", "egnn"],
            default=["gcn", "egnn"]
        )
        
        if st.button("Run Comparison"):
            comparison_results = {}
            
            for model_name in models_to_compare:
                # Create model
                model = PhysicsGNN(
                    input_dim=5,
                    hidden_dim=hidden_dim,
                    output_dim=2,
                    num_layers=num_layers,
                    model_type=model_name,
                    dropout=dropout
                )
                
                # Create test system
                test_system = ParticleSystem(
                    num_particles=num_particles,
                    dim=2,
                    interaction_type=interaction_type,
                    dt=dt,
                    gravity=gravity,
                    damping=damping,
                    seed=seed
                )
                
                # Run simulation
                positions = []
                velocities = []
                energies = []
                
                for _ in range(num_steps):
                    positions.append(test_system.positions.clone())
                    velocities.append(test_system.velocities.clone())
                    kinetic_energy = 0.5 * torch.sum(test_system.masses.unsqueeze(-1) * test_system.velocities**2)
                    energies.append(kinetic_energy.item())
                    test_system.step()
                
                comparison_results[model_name] = {
                    'positions': positions,
                    'velocities': velocities,
                    'energies': energies
                }
            
            # Plot comparison
            fig = go.Figure()
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            for i, (model_name, results) in enumerate(comparison_results.items()):
                energies = results['energies']
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(energies))),
                        y=energies,
                        mode='lines',
                        name=f'{model_name.upper()}',
                        line=dict(color=colors[i % len(colors)])
                    )
                )
            
            fig.update_layout(
                title="Energy Evolution Comparison",
                xaxis_title="Time Step",
                yaxis_title="Kinetic Energy",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model statistics
            st.subheader("Model Statistics")
            stats_data = []
            for model_name, results in comparison_results.items():
                final_energy = results['energies'][-1]
                energy_std = np.std(results['energies'])
                final_positions = results['positions'][-1].numpy()
                position_spread = np.std(final_positions)
                
                stats_data.append({
                    'Model': model_name.upper(),
                    'Final Energy': f"{final_energy:.2f}",
                    'Energy Stability': f"{energy_std:.2f}",
                    'Position Spread': f"{position_spread:.2f}"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
    
    with tab3:
        st.header("Analysis Tools")
        
        # Parameter sensitivity analysis
        st.subheader("Parameter Sensitivity")
        
        param_to_analyze = st.selectbox(
            "Parameter to analyze",
            ["gravity", "damping", "dt", "num_particles"]
        )
        
        param_values = {
            "gravity": np.linspace(1.0, 20.0, 10),
            "damping": np.linspace(0.8, 1.0, 10),
            "dt": np.linspace(0.001, 0.1, 10),
            "num_particles": np.linspace(10, 100, 10).astype(int)
        }
        
        if st.button("Run Sensitivity Analysis"):
            sensitivity_results = []
            
            for value in param_values[param_to_analyze]:
                # Create system with modified parameter
                system_params = {
                    "num_particles": num_particles,
                    "dim": 2,
                    "interaction_type": interaction_type,
                    "dt": dt,
                    "gravity": gravity,
                    "damping": damping,
                    "seed": seed
                }
                system_params[param_to_analyze] = value
                
                system = ParticleSystem(**system_params)
                
                # Run short simulation
                energies = []
                for _ in range(20):
                    kinetic_energy = 0.5 * torch.sum(system.masses.unsqueeze(-1) * system.velocities**2)
                    energies.append(kinetic_energy.item())
                    system.step()
                
                sensitivity_results.append({
                    'parameter_value': value,
                    'final_energy': energies[-1],
                    'energy_std': np.std(energies)
                })
            
            # Plot sensitivity
            sensitivity_df = pd.DataFrame(sensitivity_results)
            
            fig = px.scatter(
                sensitivity_df,
                x='parameter_value',
                y='final_energy',
                title=f"Final Energy vs {param_to_analyze.title()}",
                labels={'parameter_value': param_to_analyze.title(), 'final_energy': 'Final Energy'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("About This Demo")
        
        st.markdown("""
        ## Physics Simulation with Graph Neural Networks
        
        This interactive demo showcases the use of Graph Neural Networks (GNNs) for physics simulation.
        
        ### Features:
        - **Live Simulation**: Real-time visualization of particle dynamics
        - **Model Comparison**: Compare different GNN architectures (GCN, GAT, EGNN)
        - **Parameter Tuning**: Interactive adjustment of physics and model parameters
        - **Analysis Tools**: Sensitivity analysis and performance metrics
        
        ### Models:
        - **GCN**: Graph Convolutional Network - basic message passing
        - **GAT**: Graph Attention Network - attention-based aggregation
        - **EGNN**: E(n) Equivariant GNN - rotation/translation invariant
        
        ### Physics Types:
        - **Gravitational**: Particles attract each other
        - **Spring**: Particles connected by springs
        - **Repulsive**: Particles repel each other
        
        ### Key Metrics:
        - **Energy Conservation**: How well energy is preserved
        - **Position Stability**: Spread of particle positions
        - **Velocity Stability**: Consistency of particle velocities
        
        This demo is part of a comprehensive GNN project for physics simulations.
        """)


if __name__ == "__main__":
    main()
