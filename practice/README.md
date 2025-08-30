# Deep Learning Practice Assignments

CS231n-style practice assignments for implementing core deep learning techniques using PyTorch.

## Setup

```bash
cd practice
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run individual assignments:
```bash
python run_assignment.py activation_functions
python run_assignment.py positional_embeddings
python run_assignment.py attention_mechanisms
```

Run all assignments:
```bash
python run_assignment.py all
```

## Assignment Structure

Each assignment follows the CS231n format:
- **skeleton.py**: Your implementation goes here (fill in the TODOs)
- **test.py**: Test suite comparing your implementation with PyTorch reference
- **reference.py**: Reference implementation using PyTorch
- **README.md**: Assignment instructions and hints

## Current Assignments

1. **Activation Functions** - Implement forward/backward passes for ReLU, Sigmoid, Tanh, Swish, GELU, ELU
2. **Positional Embeddings** - Implement Sinusoidal, RoPE, T5 Relative, ALiBi positional encodings
3. **Attention Mechanisms** - Implement SDPA, MHA, MQA, GQA, MLA attention variants

## Tips

- Read the docstrings carefully - they contain important implementation details
- Use PyTorch tensor operations: `torch.matmul()`, `torch.where()`, `F.softmax()`
- Check tensor shapes frequently using `.shape`
- Start with basic functions (SDPA) before complex variants (MLA)
- Run tests frequently to catch bugs early
- Use `torch.manual_seed()` for reproducible debugging