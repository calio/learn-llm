# Assignment 2: Positional Embeddings

Implement various positional encoding schemes used in transformer architectures.

## Overview

You'll implement the following positional embedding techniques:
- Sinusoidal Positional Encoding (original Transformer)
- Learnable/Absolute Positional Embeddings
- Rotary Position Embedding (RoPE) - used in LLaMA
- Relative Position Representation (T5-style)
- ALiBi (Attention with Linear Biases) - used in BLOOM

## Files

- `skeleton.py` - **Your implementation goes here**
- `reference.py` - PyTorch reference implementation  
## Instructions

1. Open `skeleton.py` and implement the TODO sections
2. Each function should handle variable sequence lengths and embedding dimensions
3. Pay attention to the mathematical formulations in the docstrings
4. Run `python ../../run_assignment.py positional_embeddings` to check your implementation

## Tips

- Use PyTorch broadcasting effectively: `torch.arange()`, `torch.outer()`
- Be careful with tensor shapes and data types (`.long()` for indices)
- RoPE requires understanding of complex number rotations via even/odd pairs
- ALiBi uses geometric progressions for attention head slopes
- Test edge cases like sequence length = 1, even/odd dimensions

## Mathematical Notes

### Sinusoidal Encoding
For position `pos` and dimension `i`:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))  
```

### RoPE (Rotary Position Embedding)
Applies rotation matrices to embed relative positions:
```
f(x_m, m) = R_Θ,m x_m where R is rotation matrix with angle Θ_i * m
```

### Relative Position (T5)
Maps relative distances to learned embeddings with clipping:
```
relative_position = clip(i - j, -max_distance, max_distance)
```

Run your implementation with: `python ../../run_assignment.py positional_embeddings`