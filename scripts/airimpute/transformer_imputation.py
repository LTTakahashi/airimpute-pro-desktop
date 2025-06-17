#!/usr/bin/env python3
"""
Transformer-based imputation for time series data using state-of-the-art architectures.

This module implements cutting-edge transformer models for air quality data imputation,
based on recent research like ImputeFormer (Nie et al., 2023).

Complexity Analysis:
- Time: O(n²·d) where n is sequence length and d is embedding dimension
- Space: O(n²) for attention matrix storage
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for transformer imputation model."""
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512
    use_low_rank: bool = True
    rank: int = 8
    context_length: int = 96
    prediction_length: int = 24


class LowRankMultiHeadAttention(nn.Module):
    """
    Low-rank multi-head attention mechanism for efficient computation.
    
    Based on ImputeFormer's low-rankness exploitation to improve
    signal-noise balance in attention computation.
    """
    
    def __init__(self, d_model: int, n_heads: int, rank: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rank = rank
        
        # Low-rank projection layers
        self.W_q_down = nn.Linear(d_model, rank * n_heads)
        self.W_q_up = nn.Linear(rank * n_heads, d_model)
        
        self.W_k_down = nn.Linear(d_model, rank * n_heads)
        self.W_k_up = nn.Linear(rank * n_heads, d_model)
        
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with low-rank attention computation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Low-rank projections for Q and K
        Q = self.W_q_up(self.W_q_down(x))  # (batch, seq_len, d_model)
        K = self.W_k_up(self.W_k_down(x))  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.W_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + self.dropout(output))
        
        return output


class ImputeTransformerBlock(nn.Module):
    """Single transformer block with low-rank attention and feed-forward network."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        if config.use_low_rank:
            self.attention = LowRankMultiHeadAttention(
                config.d_model, config.n_heads, config.rank, config.dropout
            )
        else:
            self.attention = nn.MultiheadAttention(
                config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
            )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.layer_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Attention
        if isinstance(self.attention, LowRankMultiHeadAttention):
            attn_output = self.attention(x, mask)
        else:
            attn_output, _ = self.attention(x, x, x, attn_mask=mask)
            attn_output = self.layer_norm(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(attn_output)
        output = self.layer_norm(attn_output + ff_output)
        
        return output


class ImputeTransformer(nn.Module):
    """
    Transformer model for time series imputation.
    
    Architecture based on ImputeFormer with enhancements:
    - Low-rank attention for better signal-noise separation
    - Temporal encoding for time series awareness
    - Masked prediction for missing value imputation
    """
    
    def __init__(self, config: TransformerConfig, n_features: int):
        super().__init__()
        self.config = config
        self.n_features = n_features
        
        # Input projection
        self.input_projection = nn.Linear(n_features, config.d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(
            config.max_seq_len, config.d_model
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ImputeTransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, n_features)
        
        # Missing value token (learnable)
        self.missing_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for imputation.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_features)
            mask: Binary mask indicating missing values (1 = missing, 0 = observed)
            
        Returns:
            Imputed tensor of shape (batch, seq_len, n_features)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Project input
        x_proj = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Replace missing values with learnable token
        missing_mask = mask.unsqueeze(-1).expand(-1, -1, self.config.d_model)
        missing_tokens = self.missing_token.expand(batch_size, seq_len, -1)
        x_proj = torch.where(missing_mask.bool(), missing_tokens, x_proj)
        
        # Add positional encoding
        pe = self.positional_encoding[:, :seq_len, :].to(device)
        x_proj = x_proj + pe
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x_proj = block(x_proj)
        
        # Project back to original dimension
        output = self.output_projection(x_proj)
        
        # Only update missing values
        imputed = torch.where(mask.bool(), output, x)
        
        return imputed


class TransformerImputation:
    """
    High-level interface for transformer-based imputation.
    
    Provides training and inference capabilities with GPU support.
    """
    
    def __init__(self, config: Optional[TransformerConfig] = None, use_gpu: bool = True):
        self.config = config or TransformerConfig()
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        
        logger.info(f"Initialized TransformerImputation on device: {self.device}")
    
    def _normalize_data(self, data: np.ndarray) -> torch.Tensor:
        """Normalize data to zero mean and unit variance."""
        if self.scaler_mean is None:
            # Compute statistics only on observed values
            mask = ~np.isnan(data)
            self.scaler_mean = np.nanmean(data, axis=(0, 1), keepdims=True)
            self.scaler_std = np.nanstd(data, axis=(0, 1), keepdims=True)
            self.scaler_std[self.scaler_std == 0] = 1.0
        
        normalized = (data - self.scaler_mean) / self.scaler_std
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        return torch.FloatTensor(normalized)
    
    def _denormalize_data(self, data: torch.Tensor) -> np.ndarray:
        """Denormalize data back to original scale."""
        denormalized = data.cpu().numpy() * self.scaler_std + self.scaler_mean
        return denormalized
    
    def fit(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32,
            learning_rate: float = 1e-3, validation_split: float = 0.1) -> Dict[str, Any]:
        """
        Train the transformer model on the data.
        
        Args:
            data: Input data of shape (n_samples, seq_len, n_features)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary containing training history
        """
        n_samples, seq_len, n_features = data.shape
        
        # Initialize model
        self.model = ImputeTransformer(self.config, n_features).to(self.device)
        
        # Prepare data
        data_tensor = self._normalize_data(data).to(self.device)
        mask = torch.isnan(torch.FloatTensor(data)).to(self.device)
        
        # Split data
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            data_tensor[train_indices], mask[train_indices]
        )
        val_dataset = torch.utils.data.TensorDataset(
            data_tensor[val_indices], mask[val_indices]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        history = {"train_loss": [], "val_loss": []}
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_data, batch_mask in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                imputed = self.model(batch_data, batch_mask)
                
                # Compute loss only on missing values
                loss = F.mse_loss(
                    imputed[batch_mask.bool()],
                    batch_data[batch_mask.bool()]
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_mask in val_loader:
                    imputed = self.model(batch_data, batch_mask)
                    loss = F.mse_loss(
                        imputed[batch_mask.bool()],
                        batch_data[batch_mask.bool()]
                    )
                    val_loss += loss.item()
            
            # Update learning rate
            scheduler.step()
            
            # Record history
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - "
                          f"Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}")
        
        return history
    
    def impute(self, data: np.ndarray) -> np.ndarray:
        """
        Impute missing values in the data.
        
        Args:
            data: Input data with missing values (NaN)
            
        Returns:
            Imputed data array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        
        # Handle different input shapes
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # Add batch dimension
        
        # Prepare data
        data_tensor = self._normalize_data(data).to(self.device)
        mask = torch.isnan(torch.FloatTensor(data)).to(self.device)
        
        # Impute
        with torch.no_grad():
            imputed_normalized = self.model(data_tensor, mask)
        
        # Denormalize
        imputed = self._denormalize_data(imputed_normalized)
        
        # Remove batch dimension if added
        if imputed.shape[0] == 1:
            imputed = imputed[0]
        
        return imputed
    
    def save(self, path: str):
        """Save model and scalers to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': self.config,
            'scaler_mean': self.scaler_mean,
            'scaler_std': self.scaler_std
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and scalers from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.scaler_mean = checkpoint['scaler_mean']
        self.scaler_std = checkpoint['scaler_std']
        
        if checkpoint['model_state_dict']:
            n_features = self.scaler_mean.shape[-1]
            self.model = ImputeTransformer(self.config, n_features).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        
        logger.info(f"Model loaded from {path}")


def test_transformer_imputation():
    """Test the transformer imputation implementation."""
    # Generate synthetic data with missing values
    np.random.seed(42)
    n_samples, seq_len, n_features = 1000, 96, 5
    
    # Generate time series data
    t = np.linspace(0, 10, seq_len)
    data = np.zeros((n_samples, seq_len, n_features))
    
    for i in range(n_samples):
        for j in range(n_features):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            noise = np.random.normal(0, 0.1, seq_len)
            data[i, :, j] = np.sin(freq * t + phase) + noise
    
    # Introduce missing values (30% missing)
    mask = np.random.random((n_samples, seq_len, n_features)) < 0.3
    data[mask] = np.nan
    
    # Test imputation
    imputer = TransformerImputation(use_gpu=torch.cuda.is_available())
    
    print("Training transformer model...")
    history = imputer.fit(data, epochs=50, batch_size=32)
    
    print("Imputing missing values...")
    imputed = imputer.impute(data[:10])  # Impute first 10 samples
    
    # Calculate imputation error on known values
    known_mask = ~np.isnan(data[:10])
    mse = np.mean((data[:10][known_mask] - imputed[known_mask]) ** 2)
    print(f"MSE on known values: {mse:.4f}")
    
    # Check that all missing values were filled
    assert not np.any(np.isnan(imputed)), "Imputed data still contains NaN values"
    print("All missing values successfully imputed!")
    
    return imputer, history


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    imputer, history = test_transformer_imputation()
    
    # Print final losses
    print(f"\nFinal training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")