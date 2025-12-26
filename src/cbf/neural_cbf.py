"""
Lipschitz-Constrained Neural Control Barrier Function
========================================================

This module implements a neural network with enforced Lipschitz bounds using
spectral normalization, suitable for learning Control Barrier Functions (CBFs)
with formal safety guarantees.

Author: Claudio Urrea
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralNormLinear(nn.Module):
    """Linear layer with spectral normalization to enforce Lipschitz bound."""
    
    def __init__(self, in_features, out_features, lip_bound=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lip_bound = lip_bound
        
        # Initialize weight
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
        # Register buffer for singular value tracking
        self.register_buffer('sigma', torch.ones(1))
    
    def forward(self, x):
        """Forward pass with spectral normalization."""
        # Compute weight with spectral normalization
        W = self.get_normalized_weight()
        return F.linear(x, W, self.bias)
    
    def get_normalized_weight(self):
        """Apply spectral normalization to weight matrix."""
        # Compute largest singular value via power iteration
        sigma = self.compute_spectral_norm(self.weight)
        
        # Normalize if exceeds bound
        if sigma > self.lip_bound:
            W_normalized = self.weight * (self.lip_bound / sigma)
        else:
            W_normalized = self.weight
        
        # Update tracked sigma
        self.sigma.copy_(sigma.detach())
        
        return W_normalized
    
    @staticmethod
    def compute_spectral_norm(weight, n_iter=1):
        """Compute spectral norm (largest singular value) via power iteration."""
        with torch.no_grad():
            # Use SVD for accurate computation (slower but exact)
            _, s, _ = torch.svd(weight)
            return s[0]
    
    def enforce_lipschitz(self):
        """Explicitly enforce Lipschitz constraint (called after optimizer step)."""
        with torch.no_grad():
            sigma = self.compute_spectral_norm(self.weight)
            if sigma > self.lip_bound:
                self.weight.copy_(self.weight * (self.lip_bound / sigma))


class LipschitzNeuralCBF(nn.Module):
    """
    Lipschitz-Constrained Neural Network for Control Barrier Functions.
    
    Architecture:
        Input (66) -> Dense(128) -> tanh -> Dense(64) -> tanh -> Dense(32) -> tanh -> Output (1)
    
    All layers use spectral normalization to enforce global Lipschitz bound L.
    
    Parameters:
    -----------
    input_dim : int
        Dimension of input state (default: 66 for robot+human)
    hidden_dims : list of int
        Hidden layer dimensions (default: [128, 64, 32])
    output_dim : int
        Output dimension (default: 1 for scalar barrier)
    lipschitz_bound : float
        Global Lipschitz constant bound L (default: 1.0)
    activation : str
        Activation function ('tanh', 'relu', 'elu')
    """
    
    def __init__(
        self,
        input_dim=66,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        lipschitz_bound=1.0,
        activation='tanh'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.lipschitz_bound = lipschitz_bound
        
        # Select activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input -> first hidden
        self.layers.append(
            SpectralNormLinear(input_dim, hidden_dims[0], lipschitz_bound)
        )
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(
                SpectralNormLinear(hidden_dims[i], hidden_dims[i+1], lipschitz_bound)
            )
        
        # Last hidden -> output
        self.layers.append(
            SpectralNormLinear(hidden_dims[-1], output_dim, lipschitz_bound)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input state
        
        Returns:
        --------
        h : torch.Tensor, shape (batch_size, output_dim)
            Barrier function value
        """
        # Hidden layers with activation
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        # Output layer (linear, no activation)
        h = self.layers[-1](x)
        
        return h
    
    def enforce_lipschitz(self):
        """Enforce Lipschitz constraint on all layers."""
        for layer in self.layers:
            layer.enforce_lipschitz()
    
    def get_lipschitz_constant(self):
        """
        Compute global Lipschitz constant.
        
        Returns:
        --------
        L : float
            Upper bound on Lipschitz constant (product of layer norms)
        """
        L = 1.0
        for layer in self.layers:
            L *= layer.sigma.item()
        
        # Account for activation Lipschitz constant (tanh: 1, relu: 1, elu: 1)
        # For tanh: |tanh(a) - tanh(b)| <= |a - b|
        
        return L
    
    def compute_gradient(self, x):
        """
        Compute gradient ∇h(x).
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input state
        
        Returns:
        --------
        grad_h : torch.Tensor, shape (batch_size, input_dim)
            Gradient of barrier function
        """
        x = x.requires_grad_(True)
        h = self.forward(x)
        
        grad_h = torch.autograd.grad(
            outputs=h,
            inputs=x,
            grad_outputs=torch.ones_like(h),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return grad_h
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'lipschitz_bound': self.lipschitz_bound
        }, path)
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            output_dim=checkpoint['output_dim'],
            lipschitz_bound=checkpoint['lipschitz_bound']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        return model
    
    def __repr__(self):
        return (
            f"LipschitzNeuralCBF(\n"
            f"  Architecture: {self.input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> {self.output_dim}\n"
            f"  Lipschitz bound: L ≤ {self.lipschitz_bound}\n"
            f"  Activation: {self.activation.__name__}\n"
            f"  Parameters: {sum(p.numel() for p in self.parameters())}\n"
            f")"
        )


# ========================================
# Utility Functions
# ========================================

def test_lipschitz_property(model, n_samples=1000, input_dim=66):
    """
    Empirically test Lipschitz property of the neural network.
    
    Parameters:
    -----------
    model : LipschitzNeuralCBF
        Trained model
    n_samples : int
        Number of random sample pairs to test
    input_dim : int
        Input dimension
    
    Returns:
    --------
    L_empirical : float
        Empirical Lipschitz constant estimate
    """
    model.eval()
    
    with torch.no_grad():
        max_ratio = 0.0
        
        for _ in range(n_samples):
            # Random pairs
            x1 = torch.randn(1, input_dim)
            x2 = torch.randn(1, input_dim)
            
            # Compute outputs
            h1 = model(x1)
            h2 = model(x2)
            
            # Compute ratio |h(x1) - h(x2)| / |x1 - x2|
            numerator = torch.abs(h1 - h2).item()
            denominator = torch.norm(x1 - x2).item()
            
            if denominator > 1e-6:
                ratio = numerator / denominator
                max_ratio = max(max_ratio, ratio)
        
        return max_ratio


if __name__ == '__main__':
    # Example usage
    print("Testing Lipschitz Neural CBF")
    print("="*60)
    
    # Create model
    model = LipschitzNeuralCBF(
        input_dim=66,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        lipschitz_bound=1.0,
        activation='tanh'
    )
    
    print(model)
    print()
    
    # Test forward pass
    x_test = torch.randn(10, 66)
    h_test = model(x_test)
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {h_test.shape}")
    print(f"Theoretical Lipschitz: L ≤ {model.lipschitz_bound}")
    print(f"Current Lipschitz: L ≈ {model.get_lipschitz_constant():.4f}")
    print()
    
    # Test empirical Lipschitz
    L_empirical = test_lipschitz_property(model, n_samples=1000)
    print(f"Empirical Lipschitz (1000 samples): L ≈ {L_empirical:.4f}")
    print()
    
    # Test gradient computation
    grad_h = model.compute_gradient(x_test)
    print(f"Gradient shape: {grad_h.shape}")
    print(f"Gradient norm (avg): {grad_h.norm(dim=1).mean().item():.4f}")
    
    print("="*60)
    print("Test completed successfully!")
