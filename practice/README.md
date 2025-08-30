# Deep Learning Practice Assignments

CS231n-style practice assignments for implementing core deep learning techniques from scratch.

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

1. **Activation Functions** - Implement forward/backward passes for common activation functions
2. **Positional Embeddings** - Implement various positional encoding schemes

## Tips

- Read the docstrings carefully - they contain important implementation details
- Start with the forward pass, then implement the backward pass
- Check shapes and numerical stability
- Run tests frequently to catch bugs early