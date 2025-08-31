# Assignment 1: Activation Functions

Implement forward and backward passes for common activation functions used in deep learning.

## Overview

You'll implement the following activation functions from scratch:
- ReLU and its variants (Leaky ReLU, ELU)
- Sigmoid and Tanh
- Swish/SiLU
- GELU (both exact and approximation)

## Files

- `skeleton.py` - **Your implementation goes here**
- `reference.py` - PyTorch reference implementation
## Instructions

1. Open `skeleton.py` and implement the TODO sections
2. Each function should return both the forward pass result and cache for backward pass
3. Implement corresponding backward pass functions
4. Run `python ../../run_assignment.py activation_functions` to check your implementation

## Tips

- Pay attention to numerical stability (especially for sigmoid/tanh)
- Handle edge cases (like x=0 for leaky ReLU)
- Use the chain rule for backward passes
- Check the shapes of your gradients
- Some functions have analytical derivatives, others require the chain rule

## Mathematical Notes

### ReLU
- Forward: `f(x) = max(0, x)`
- Backward: `df/dx = 1 if x > 0 else 0`

### Sigmoid  
- Forward: `f(x) = 1 / (1 + exp(-x))`
- Backward: `df/dx = f(x) * (1 - f(x))`

### GELU (exact)
- Forward: `f(x) = x * Φ(x)` where Φ is the standard normal CDF
- Can be computed using the error function: `Φ(x) = 0.5 * (1 + erf(x/√2))`

Run your implementation with: `python ../../run_assignment.py activation_functions`