#!/usr/bin/env python3
"""
Train Neural Control Barrier Function
==========================================

This script trains a Lipschitz-constrained neural network to learn a Control 
Barrier Function (CBF) from simulation data.

Usage:
    python scripts/train_cbf.py --samples 50000 --epochs 500 --output models/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cbf.neural_cbf import LipschitzNeuralCBF
from src.utils.data_generation import generate_training_data
from src.cbf.validation import validate_cbf_properties


def main(args):
    """Main training loop."""
    
    print("="*60)
    print("LBCF Training Pipeline")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")
    
    # ========================================
    # Step 1: Generate or Load Training Data
    # ========================================
    
    print("\n[1/5] Generating training data...")
    
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading existing data from {args.data_path}")
        data = torch.load(args.data_path)
        X_train, y_train = data['X'], data['y']
    else:
        print(f"Generating {args.samples} new samples...")
        X_train, y_train = generate_training_data(
            n_samples=args.samples,
            safe_ratio=0.4,
            unsafe_ratio=0.4,
            boundary_ratio=0.2,
            seed=args.seed
        )
        
        # Save generated data
        if args.save_data:
            data_dir = Path("data/training")
            data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {'X': X_train, 'y': y_train},
                data_dir / f"training_data_{args.samples}.pt"
            )
    
    # Create DataLoader
    dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    print(f"Training set size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    
    # ========================================
    # Step 2: Initialize Neural CBF
    # ========================================
    
    print("\n[2/5] Initializing neural network...")
    
    model = LipschitzNeuralCBF(
        input_dim=66,
        hidden_dims=[128, 64, 32],
        output_dim=1,
        lipschitz_bound=args.lipschitz_bound,
        activation='tanh'
    ).to(device)
    
    print(f"Model architecture: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Lipschitz bound: L ≤ {args.lipschitz_bound}")
    
    # ========================================
    # Step 3: Setup Training
    # ========================================
    
    print("\n[3/5] Setting up training...")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95
    )
    
    # Loss weights
    loss_weights = {
        'safety': 10.0,
        'validity': 5.0,
        'smooth': 0.1,
        'cbf': 2.0
    }
    
    print(f"Optimizer: Adam (lr={args.lr})")
    print(f"Scheduler: ExponentialLR (gamma=0.95)")
    print(f"Loss weights: {loss_weights}")
    
    # ========================================
    # Step 4: Training Loop
    # ========================================
    
    print("\n[4/5] Training...")
    
    history = {
        'train_loss': [],
        'safety_loss': [],
        'validity_loss': [],
        'smooth_loss': [],
        'cbf_loss': [],
        'lr': []
    }
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {key: 0.0 for key in history.keys() if key != 'lr'}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            h_pred = model(x_batch)
            
            # Compute losses
            loss_safety = compute_safety_loss(h_pred, y_batch)
            loss_validity = compute_validity_loss(model, x_batch, h_pred)
            loss_smooth = compute_smoothness_loss(model, x_batch)
            loss_cbf = compute_cbf_decrease_loss(h_pred)
            
            # Total loss
            total_loss = (
                loss_weights['safety'] * loss_safety +
                loss_weights['validity'] * loss_validity +
                loss_weights['smooth'] * loss_smooth +
                loss_weights['cbf'] * loss_cbf
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Enforce Lipschitz constraint
            model.enforce_lipschitz()
            
            # Accumulate losses
            epoch_losses['train_loss'] += total_loss.item()
            epoch_losses['safety_loss'] += loss_safety.item()
            epoch_losses['validity_loss'] += loss_validity.item()
            epoch_losses['smooth_loss'] += loss_smooth.item()
            epoch_losses['cbf_loss'] += loss_cbf.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'safety': f"{loss_safety.item():.4f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
            history[key].append(epoch_losses[key])
        
        history['lr'].append(scheduler.get_last_lr()[0])
        
        # Learning rate decay every 100 epochs
        if (epoch + 1) % 100 == 0:
            scheduler.step()
        
        # Print epoch summary
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Loss: {epoch_losses['train_loss']:.4f}")
            print(f"  Safety: {epoch_losses['safety_loss']:.4f}")
            print(f"  Validity: {epoch_losses['validity_loss']:.4f}")
            print(f"  LR: {history['lr'][-1]:.6f}")
        
        # Save best model
        if epoch_losses['train_loss'] < best_loss:
            best_loss = epoch_losses['train_loss']
            save_checkpoint(model, optimizer, epoch, args.output_dir, 'best')
    
    # ========================================
    # Step 5: Validation and Save
    # ========================================
    
    print("\n[5/5] Validation and saving...")
    
    # Validate CBF properties
    validation_results = validate_cbf_properties(
        model,
        X_train,
        y_train,
        device=device
    )
    
    print("\nValidation Results:")
    print(f"  Approximation error σ: {validation_results['sigma']:.4f} m")
    print(f"  Lipschitz constant L: {validation_results['lipschitz']:.4f}")
    print(f"  CBF validity rate: {validation_results['validity_rate']:.2%}")
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, args.output_dir, 'final')
    
    # Save metadata
    metadata = {
        'architecture': [66, 128, 64, 32, 1],
        'lipschitz_bound': args.lipschitz_bound,
        'training_samples': args.samples,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'validation': validation_results,
        'history': history
    }
    
    with open(os.path.join(args.output_dir, 'cbf_model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\nTraining complete!")
    print(f"Models saved to: {args.output_dir}")
    print("="*60)


def compute_safety_loss(h_pred, y_true):
    """Safety classification loss."""
    loss_safe = torch.relu(-h_pred[y_true > 0]).mean()
    loss_unsafe = torch.relu(h_pred[y_true < 0]).mean()
    return loss_safe + loss_unsafe


def compute_validity_loss(model, x, h_pred):
    """CBF validity condition loss."""
    # Simplified: check gradient magnitude
    x.requires_grad_(True)
    grad_h = torch.autograd.grad(
        h_pred.sum(), x, create_graph=True
    )[0]
    
    # Penalize small gradients (flat regions)
    validity_loss = torch.relu(0.01 - grad_h.norm(dim=1)).mean()
    return validity_loss


def compute_smoothness_loss(model, x):
    """Smoothness regularization."""
    x.requires_grad_(True)
    h = model(x)
    grad_h = torch.autograd.grad(
        h.sum(), x, create_graph=True
    )[0]
    return (grad_h ** 2).mean()


def compute_cbf_decrease_loss(h_pred):
    """Penalize barrier decrease (simplified)."""
    # In practice, this requires trajectory pairs
    # Here we use a placeholder
    return torch.tensor(0.0, device=h_pred.device)


def save_checkpoint(model, optimizer, epoch, output_dir, name):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    path = os.path.join(output_dir, f'cbf_model_{name}.pth')
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Lipschitz-constrained Neural CBF'
    )
    
    # Data arguments
    parser.add_argument('--samples', type=int, default=50000,
                        help='Number of training samples')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to existing training data')
    parser.add_argument('--save-data', action='store_true',
                        help='Save generated training data')
    
    # Model arguments
    parser.add_argument('--lipschitz-bound', type=float, default=1.0,
                        help='Lipschitz constant bound')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    
    # System arguments
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models/',
                        help='Output directory for models')
    
    args = parser.parse_args()
    
    main(args)
