"""
Nonlinear Least Squares Estimation using PyTorch

This script demonstrates:
1. Defining a nonlinear model with known parameters
2. Generating synthetic training data from the model
3. Using PyTorch optimization to estimate the original parameters
4. Validating that learned parameters match the true parameters
"""

import torch
import torch.nn as nn
import numpy as np


class NonlinearModel(nn.Module):
    """
    A nonlinear model: y = a * sin(b * x) + c * x + d
    
    This represents a combination of sinusoidal and linear terms,
    commonly found in signal processing and physics applications.
    """
    
    def __init__(self, a=1.0, b=1.0, c=1.0, d=1.0):
        super().__init__()
        # Initialize parameters as learnable
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
        self.d = nn.Parameter(torch.tensor(d, dtype=torch.float32))
    
    def forward(self, x):
        """Compute y = a * sin(b * x) + c * x + d"""
        return self.a * torch.sin(self.b * x) + self.c * x + self.d


def generate_samples(true_model, num_samples=100, x_range=(-1, 1), noise_std=0.1, device='cpu'):
    """
    Generate synthetic training data from the true model.
    
    Args:
        true_model: The model with true parameters
        num_samples: Number of data points to generate
        x_range: Range for input values (min, max)
        noise_std: Standard deviation of Gaussian noise added to outputs
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        x: Input tensor of shape (num_samples, 1)
        y: Output tensor of shape (num_samples, 1)
    """
    # Generate random input values
    x = torch.linspace(x_range[0], x_range[1], num_samples, device=device).unsqueeze(1)
    
    # Generate outputs using the true model
    with torch.no_grad():
        y_true = true_model(x)
    
    # Add Gaussian noise
    noise = torch.randn_like(y_true) * noise_std
    y = y_true + noise
    
    return x, y


def train_model(model, x_train, y_train, learning_rate=0.01, num_epochs=1000, device='cpu'):
    """
    Train the model to estimate parameters using gradient descent.
    
    Args:
        model: The model to train
        x_train: Training input data
        y_train: Training output data
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        losses: List of loss values over epochs
    """
    # Use MSE loss (equivalent to least squares)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
    
    return losses


def main():
    """Main function to demonstrate the complete workflow."""
    
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define true model parameters
    true_params = {'a': 2.5, 'b': 3.0, 'c': 0.5, 'd': 1.0}
    print("\n" + "="*60)
    print("TRUE MODEL PARAMETERS:")
    print("="*60)
    for param, value in true_params.items():
        print(f"  {param} = {value:.4f}")
    
    # Create the true model
    true_model = NonlinearModel(**true_params).to(device)
    
    # Generate synthetic training data
    print("\n" + "="*60)
    print("GENERATING SYNTHETIC DATA:")
    print("="*60)
    num_samples = 500
    x_train, y_train = generate_samples(
        true_model, 
        num_samples=num_samples,
        x_range=(-2, 2),
        noise_std=0.02,
        device=device
    )
    print(f"  Generated {num_samples} training samples")
    print(f"  Input range: [{x_train.min().item():.2f}, {x_train.max().item():.2f}]")
    print(f"  Output range: [{y_train.min().item():.2f}, {y_train.max().item():.2f}]")
    
    # Create a new model with random initialization for learning
    learned_model = NonlinearModel(a=1.0, b=1.0, c=0.0, d=0.0).to(device)
    
    print("\n" + "="*60)
    print("INITIAL LEARNED MODEL PARAMETERS (before training):")
    print("="*60)
    with torch.no_grad():
        print(f"  a = {learned_model.a.item():.4f}")
        print(f"  b = {learned_model.b.item():.4f}")
        print(f"  c = {learned_model.c.item():.4f}")
        print(f"  d = {learned_model.d.item():.4f}")
    
    # Train the model
    print("\n" + "="*60)
    print("TRAINING MODEL:")
    print("="*60)
    losses = train_model(
        learned_model, 
        x_train, 
        y_train, 
        learning_rate=0.1,
        num_epochs=2000,
        device=device
    )
    
    # Display final learned parameters
    print("\n" + "="*60)
    print("LEARNED MODEL PARAMETERS (after training):")
    print("="*60)
    with torch.no_grad():
        learned_a = learned_model.a.item()
        learned_b = learned_model.b.item()
        learned_c = learned_model.c.item()
        learned_d = learned_model.d.item()
        
        print(f"  a = {learned_a:.4f} (true: {true_params['a']:.4f}, error: {abs(learned_a - true_params['a']):.4f})")
        print(f"  b = {learned_b:.4f} (true: {true_params['b']:.4f}, error: {abs(learned_b - true_params['b']):.4f})")
        print(f"  c = {learned_c:.4f} (true: {true_params['c']:.4f}, error: {abs(learned_c - true_params['c']):.4f})")
        print(f"  d = {learned_d:.4f} (true: {true_params['d']:.4f}, error: {abs(learned_d - true_params['d']):.4f})")
    
    # Validate parameter recovery
    print("\n" + "="*60)
    print("VALIDATION:")
    print("="*60)
    
    tolerance = 0.1
    all_params_recovered = True
    
    for param_name, true_value in true_params.items():
        learned_value = getattr(learned_model, param_name).item()
        error = abs(learned_value - true_value)
        
        if error < tolerance:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_params_recovered = False
        
        print(f"  {param_name}: {status} (error = {error:.4f}, tolerance = {tolerance})")
    
    print("\n" + "="*60)
    if all_params_recovered:
        print("SUCCESS: All parameters successfully recovered within tolerance!")
    else:
        print("WARNING: Some parameters were not recovered within tolerance.")
        print("This may indicate insufficient training or high noise levels.")
    print("="*60)
    
    print(f"\nFinal training loss: {losses[-1]:.6f}")
    print(f"Device used: {device}")


if __name__ == "__main__":
    main()
