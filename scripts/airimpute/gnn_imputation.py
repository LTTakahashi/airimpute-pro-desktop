#!/usr/bin/env python3
"""
Graph Neural Network (GNN) based imputation for spatial data.

This module implements GNN models for air quality data imputation,
leveraging spatial relationships between monitoring stations.

Complexity Analysis:
- Time: O(n·e·h) where n is number of nodes, e is edges, h is hidden dimension
- Space: O(n·h + e) for node features and edge connections
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops, degree
from typing import Tuple, Optional, Dict, Any, List
import logging
from dataclasses import dataclass
from scipy.spatial import Delaunay
from sklearn.neighbors import kneighbors_graph
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class GNNConfig:
    """Configuration for GNN imputation model."""
    hidden_dims: List[int] = None
    n_layers: int = 3
    dropout: float = 0.1
    attention_heads: int = 8
    edge_threshold: float = 100.0  # km for spatial edges
    k_neighbors: int = 5
    use_attention: bool = True
    aggregation: str = "mean"  # mean, max, add
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 64]


class SpatialGraphConstructor:
    """
    Constructs spatial graphs from monitoring station locations.
    
    Supports multiple graph construction methods:
    - K-nearest neighbors
    - Distance threshold
    - Delaunay triangulation
    - Correlation-based edges
    """
    
    def __init__(self, method: str = "knn", **kwargs):
        self.method = method
        self.kwargs = kwargs
    
    def construct_graph(self, locations: np.ndarray, 
                       features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct graph from spatial locations.
        
        Args:
            locations: Array of shape (n_nodes, 2) with (lat, lon) coordinates
            features: Optional node features for correlation-based edges
            
        Returns:
            edge_index: Array of shape (2, n_edges) with edge connections
            edge_attr: Array of shape (n_edges, d) with edge attributes
        """
        if self.method == "knn":
            return self._knn_graph(locations)
        elif self.method == "distance":
            return self._distance_graph(locations)
        elif self.method == "delaunay":
            return self._delaunay_graph(locations)
        elif self.method == "correlation":
            if features is None:
                raise ValueError("Features required for correlation-based graph")
            return self._correlation_graph(locations, features)
        else:
            raise ValueError(f"Unknown graph construction method: {self.method}")
    
    def _knn_graph(self, locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct k-nearest neighbors graph."""
        k = self.kwargs.get("k", 5)
        
        # Compute pairwise distances
        distances = self._compute_distances(locations)
        
        # Find k-nearest neighbors
        n_nodes = len(locations)
        edge_list = []
        edge_weights = []
        
        for i in range(n_nodes):
            # Get k nearest neighbors (excluding self)
            neighbors = np.argsort(distances[i])[1:k+1]
            
            for j in neighbors:
                edge_list.append([i, j])
                edge_weights.append(1.0 / (distances[i, j] + 1e-6))
        
        edge_index = np.array(edge_list).T
        edge_attr = np.array(edge_weights).reshape(-1, 1)
        
        return edge_index, edge_attr
    
    def _distance_graph(self, locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct graph based on distance threshold."""
        threshold = self.kwargs.get("threshold", 100.0)
        
        distances = self._compute_distances(locations)
        edges = np.where((distances > 0) & (distances < threshold))
        
        edge_index = np.vstack(edges)
        edge_attr = 1.0 / (distances[edges] + 1e-6)
        edge_attr = edge_attr.reshape(-1, 1)
        
        return edge_index, edge_attr
    
    def _delaunay_graph(self, locations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct Delaunay triangulation graph."""
        tri = Delaunay(locations)
        edges = set()
        
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        
        edge_list = list(edges)
        edge_index = np.array(edge_list).T
        
        # Compute edge weights based on distances
        distances = self._compute_distances(locations)
        edge_weights = []
        for i, j in edge_list:
            edge_weights.append(1.0 / (distances[i, j] + 1e-6))
        
        edge_attr = np.array(edge_weights).reshape(-1, 1)
        
        return edge_index, edge_attr
    
    def _correlation_graph(self, locations: np.ndarray, 
                          features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Construct graph based on feature correlations."""
        threshold = self.kwargs.get("corr_threshold", 0.7)
        
        # Compute correlations
        n_nodes = len(locations)
        corr_matrix = np.corrcoef(features)
        
        # Create edges for high correlations
        edges = np.where((corr_matrix > threshold) & 
                        (np.arange(n_nodes)[:, None] != np.arange(n_nodes)))
        
        edge_index = np.vstack(edges)
        edge_attr = corr_matrix[edges].reshape(-1, 1)
        
        return edge_index, edge_attr
    
    def _compute_distances(self, locations: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between locations (in km)."""
        # Convert lat/lon to radians
        locations_rad = np.radians(locations)
        
        # Haversine distance
        lat1 = locations_rad[:, 0][:, np.newaxis]
        lat2 = locations_rad[:, 0][np.newaxis, :]
        lon1 = locations_rad[:, 1][:, np.newaxis]
        lon2 = locations_rad[:, 1][np.newaxis, :]
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        R = 6371.0
        distances = R * c
        
        return distances


class GraphAttentionLayer(MessagePassing):
    """
    Custom Graph Attention Layer for spatial imputation.
    
    Implements multi-head attention mechanism over graph edges
    with missing value awareness.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 heads: int = 8, dropout: float = 0.1):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Attention parameters
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        # Output projection
        self.out_proj = nn.Linear(heads * out_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        # Add self-loops
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, 
                                              fill_value=1.0, num_nodes=x.size(0))
        
        # Linear transformation
        x = self.W(x).view(-1, self.heads, self.out_channels)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Project output
        out = out.view(-1, self.heads * self.out_channels)
        out = self.out_proj(out)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute messages with attention weights."""
        # Compute attention scores
        x_cat = torch.cat([x_i, x_j], dim=-1)
        alpha = (x_cat * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.softmax(alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply edge weights if provided
        if edge_attr is not None:
            alpha = alpha * edge_attr.view(-1, 1)
        
        # Weight messages by attention
        return x_j * alpha.unsqueeze(-1)


class SpatialGNN(nn.Module):
    """
    Graph Neural Network for spatial imputation.
    
    Architecture supports both GCN and GAT layers with
    skip connections and missing value handling.
    """
    
    def __init__(self, config: GNNConfig, n_features: int):
        super().__init__()
        self.config = config
        self.n_features = n_features
        
        # Build GNN layers
        self.layers = nn.ModuleList()
        in_dim = n_features
        
        for i, hidden_dim in enumerate(config.hidden_dims):
            if config.use_attention:
                layer = GraphAttentionLayer(
                    in_dim, hidden_dim, 
                    heads=config.attention_heads if i < len(config.hidden_dims) - 1 else 1,
                    dropout=config.dropout
                )
            else:
                layer = GCNConv(in_dim, hidden_dim)
            
            self.layers.append(layer)
            in_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_dims[-1], n_features)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Skip connection
        self.skip_proj = nn.Linear(n_features, n_features)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for spatial imputation.
        
        Args:
            x: Node features of shape (n_nodes, n_features)
            edge_index: Edge connections of shape (2, n_edges)
            edge_attr: Optional edge attributes
            mask: Binary mask for missing values
            
        Returns:
            Imputed features of shape (n_nodes, n_features)
        """
        # Store input for skip connection
        x_input = x
        
        # Replace missing values with zeros
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), torch.zeros_like(x), x)
        
        # Pass through GNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output projection
        out = self.output_layer(x)
        
        # Skip connection
        out = out + self.skip_proj(x_input)
        
        # Only update missing values
        if mask is not None:
            out = torch.where(mask.unsqueeze(-1), out, x_input)
        
        return out


class GNNImputation:
    """
    High-level interface for GNN-based spatial imputation.
    
    Provides training and inference with automatic graph construction.
    """
    
    def __init__(self, config: Optional[GNNConfig] = None, 
                 graph_method: str = "knn",
                 use_gpu: bool = True):
        self.config = config or GNNConfig()
        self.graph_method = graph_method
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.graph_constructor = SpatialGraphConstructor(
            method=graph_method, 
            k=config.k_neighbors if config else 5,
            threshold=config.edge_threshold if config else 100.0
        )
        self.scaler_mean = None
        self.scaler_std = None
        
        logger.info(f"Initialized GNNImputation on device: {self.device}")
    
    def _normalize_features(self, features: np.ndarray) -> torch.Tensor:
        """Normalize features to zero mean and unit variance."""
        if self.scaler_mean is None:
            mask = ~np.isnan(features)
            self.scaler_mean = np.nanmean(features, axis=0, keepdims=True)
            self.scaler_std = np.nanstd(features, axis=0, keepdims=True)
            self.scaler_std[self.scaler_std == 0] = 1.0
        
        normalized = (features - self.scaler_mean) / self.scaler_std
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        return torch.FloatTensor(normalized)
    
    def _denormalize_features(self, features: torch.Tensor) -> np.ndarray:
        """Denormalize features back to original scale."""
        denormalized = features.cpu().numpy() * self.scaler_std + self.scaler_mean
        return denormalized
    
    def fit(self, features: np.ndarray, locations: np.ndarray,
            epochs: int = 100, batch_size: int = 1,
            learning_rate: float = 1e-3) -> Dict[str, Any]:
        """
        Train the GNN model.
        
        Args:
            features: Node features of shape (n_nodes, n_features) with missing values
            locations: Spatial locations of shape (n_nodes, 2) with (lat, lon)
            epochs: Number of training epochs
            batch_size: Batch size (usually 1 for full graph)
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        n_nodes, n_features = features.shape
        
        # Initialize model
        self.model = SpatialGNN(self.config, n_features).to(self.device)
        
        # Construct graph
        edge_index, edge_attr = self.graph_constructor.construct_graph(locations)
        edge_index = torch.LongTensor(edge_index).to(self.device)
        edge_attr = torch.FloatTensor(edge_attr).to(self.device)
        
        # Prepare features
        features_tensor = self._normalize_features(features).to(self.device)
        mask = torch.isnan(torch.FloatTensor(features)).to(self.device)
        
        # Create data object
        data = Data(x=features_tensor, edge_index=edge_index, edge_attr=edge_attr)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        
        history = {"loss": []}
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            out = self.model(data.x, data.edge_index, data.edge_attr, mask)
            
            # Compute loss only on observed values
            observed_mask = ~mask
            loss = F.mse_loss(out[observed_mask], features_tensor[observed_mask])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            history["loss"].append(loss.item())
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
        
        return history
    
    def impute(self, features: np.ndarray, locations: np.ndarray) -> np.ndarray:
        """
        Impute missing values using trained GNN.
        
        Args:
            features: Node features with missing values
            locations: Spatial locations
            
        Returns:
            Imputed features
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        
        # Construct graph
        edge_index, edge_attr = self.graph_constructor.construct_graph(locations)
        edge_index = torch.LongTensor(edge_index).to(self.device)
        edge_attr = torch.FloatTensor(edge_attr).to(self.device)
        
        # Prepare features
        features_tensor = self._normalize_features(features).to(self.device)
        mask = torch.isnan(torch.FloatTensor(features)).to(self.device)
        
        # Impute
        with torch.no_grad():
            imputed_normalized = self.model(
                features_tensor, edge_index, edge_attr, mask
            )
        
        # Denormalize
        imputed = self._denormalize_features(imputed_normalized)
        
        return imputed
    
    def visualize_graph(self, locations: np.ndarray, 
                       features: Optional[np.ndarray] = None) -> nx.Graph:
        """
        Create NetworkX graph for visualization.
        
        Args:
            locations: Spatial locations
            features: Optional features for node attributes
            
        Returns:
            NetworkX graph object
        """
        edge_index, edge_attr = self.graph_constructor.construct_graph(locations, features)
        
        G = nx.Graph()
        
        # Add nodes with positions
        for i, loc in enumerate(locations):
            G.add_node(i, pos=(loc[1], loc[0]))  # (lon, lat) for plotting
        
        # Add edges with weights
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            weight = edge_attr[i, 0] if edge_attr.shape[1] > 0 else 1.0
            G.add_edge(int(src), int(dst), weight=float(weight))
        
        return G


def test_gnn_imputation():
    """Test the GNN imputation implementation."""
    np.random.seed(42)
    
    # Generate synthetic monitoring stations
    n_stations = 20
    locations = np.random.uniform(-1, 1, (n_stations, 2)) * 10  # ±10 degrees
    locations[:, 0] += 40  # Center around 40°N
    locations[:, 1] += -100  # Center around 100°W
    
    # Generate correlated features based on distance
    n_features = 3
    features = np.zeros((n_stations, n_features))
    
    for f in range(n_features):
        # Generate smooth spatial field
        center = np.random.uniform(-1, 1, 2) * 5
        for i in range(n_stations):
            dist = np.linalg.norm(locations[i] - center)
            features[i, f] = np.exp(-0.1 * dist) + np.random.normal(0, 0.1)
    
    # Introduce missing values (40% missing)
    mask = np.random.random((n_stations, n_features)) < 0.4
    features[mask] = np.nan
    
    # Test different graph construction methods
    for method in ["knn", "distance", "delaunay"]:
        print(f"\nTesting {method} graph construction...")
        
        imputer = GNNImputation(
            config=GNNConfig(hidden_dims=[32, 64, 32]),
            graph_method=method,
            use_gpu=torch.cuda.is_available()
        )
        
        # Train model
        history = imputer.fit(features, locations, epochs=50)
        
        # Impute
        imputed = imputer.impute(features, locations)
        
        # Calculate error on known values
        known_mask = ~np.isnan(features)
        mse = np.mean((features[known_mask] - imputed[known_mask]) ** 2)
        print(f"MSE on known values ({method}): {mse:.4f}")
        
        # Visualize graph
        G = imputer.visualize_graph(locations)
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    assert not np.any(np.isnan(imputed)), "Imputed data contains NaN values"
    print("\nAll tests passed!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Check for required packages
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("Warning: torch_geometric not installed. Install with:")
        print("pip install torch-geometric")
    
    # Run tests
    test_gnn_imputation()