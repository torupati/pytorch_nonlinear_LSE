# pytorch_nonlinear_LSE

Test code to solve nonlinear least square error minimization using PyTorch.

## Overview

This project demonstrates a complete workflow for parameter estimation in nonlinear models using PyTorch:

1. **Model Definition**: Defines a nonlinear model with known parameters
2. **Sample Generation**: Generates synthetic training data from the model
3. **Parameter Learning**: Uses gradient descent optimization to estimate parameters
4. **Validation**: Confirms that learned parameters match the original model parameters

The current implementation uses a nonlinear model combining sinusoidal and linear terms:
```
y = a * sin(b * x) + c * x + d
```

## Requirements

- Python 3.12+
- PyTorch 2.10+
- NumPy 2.4+

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Install uv if you haven't already
pip install uv

# Install dependencies
uv sync
```

## Usage

Run the main script to see the complete parameter estimation workflow:

```bash
uv run python main.py
```

The script will:
- Display the true model parameters
- Generate synthetic training data with added noise
- Show initial (random) parameter values
- Train the model using gradient descent
- Display learned parameters and compare with true values
- Validate that all parameters are recovered within tolerance

### Example Output

```
Using device: cpu

============================================================
TRUE MODEL PARAMETERS:
============================================================
  a = 2.5000
  b = 3.0000
  c = 0.5000
  d = 1.0000

============================================================
GENERATING SYNTHETIC DATA:
============================================================
  Generated 500 training samples
  Input range: [-2.00, 2.00]
  Output range: [-1.80, 3.79]

...

============================================================
LEARNED MODEL PARAMETERS (after training):
============================================================
  a = 2.5006 (true: 2.5000, error: 0.0006)
  b = 3.0004 (true: 3.0000, error: 0.0004)
  c = 0.5000 (true: 0.5000, error: 0.0000)
  d = 1.0009 (true: 1.0000, error: 0.0009)

============================================================
VALIDATION:
============================================================
  a: ✓ PASS (error = 0.0006, tolerance = 0.1)
  b: ✓ PASS (error = 0.0004, tolerance = 0.1)
  c: ✓ PASS (error = 0.0000, tolerance = 0.1)
  d: ✓ PASS (error = 0.0009, tolerance = 0.1)

============================================================
SUCCESS: All parameters successfully recovered within tolerance!
============================================================
```

## GPU Support

The code automatically detects and uses GPU (CUDA) if available, otherwise falls back to CPU:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## Project Structure

```
pytorch_nonlinear_LSE/
├── main.py              # Main implementation and demo
├── pyproject.toml       # Project configuration and dependencies
├── README.md            # This file
└── LICENSE              # License information
```

## How It Works

### 1. Model Definition

The `NonlinearModel` class defines a PyTorch module with four learnable parameters:

```python
class NonlinearModel(nn.Module):
    def __init__(self, a=1.0, b=1.0, c=1.0, d=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
        self.d = nn.Parameter(torch.tensor(d, dtype=torch.float32))
```

### 2. Data Generation

Synthetic data is generated from the true model with added Gaussian noise:

```python
x = torch.linspace(x_range[0], x_range[1], num_samples)
y_true = true_model(x)
y = y_true + torch.randn_like(y_true) * noise_std
```

### 3. Training

The model is trained using the Adam optimizer with Mean Squared Error (MSE) loss, which is equivalent to least squares optimization:

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

### 4. Validation

After training, the learned parameters are compared against the true values to verify successful parameter recovery.

## License

See LICENSE file for details.

