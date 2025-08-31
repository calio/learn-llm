# Assignment 3: Attention Mechanisms

Implement various attention mechanisms used in modern transformer architectures.

## Overview

You'll implement the following attention variants:
- **SDPA (Scaled Dot-Product Attention)** - The core attention mechanism
- **MHA (Multi-Head Attention)** - Original transformer attention
- **MQA (Multi-Query Attention)** - Shared key/value across heads
- **GQA (Grouped-Query Attention)** - Groups of queries share key/value
- **MLA (Multi-head Latent Attention)** - DeepSeek-V2 style latent attention

## Files

- `skeleton.py` - **Your implementation goes here**
- `reference.py` - PyTorch reference implementation
## Instructions

1. Open `skeleton.py` and implement the TODO sections
2. Start with SDPA, then build up to more complex variants
3. Pay careful attention to tensor shapes and broadcasting
4. Implement both forward pass and attention weight computation
5. Run `python ../../run_assignment.py attention_mechanisms` to check your implementation

## Tips

- Understand the core SDPA formula: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- Use efficient matrix operations and broadcasting
- Handle masking properly (causal masks, padding masks)
- Be careful with numerical stability in softmax
- MQA/GQA reduce KV cache size - understand the memory benefits
- MLA introduces learnable compression of KV representations

## Mathematical Notes

### SDPA (Scaled Dot-Product Attention)
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```
- Q: [batch, seq_len, d_k] queries
- K: [batch, seq_len, d_k] keys  
- V: [batch, seq_len, d_v] values
- Output: [batch, seq_len, d_v]

### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

### Multi-Query Attention (MQA)
- Single key/value head shared across all query heads
- Reduces KV cache from `h×d_k` to `1×d_k` per position

### Grouped-Query Attention (GQA)  
- Groups of query heads share key/value heads
- Interpolates between MHA (g=h) and MQA (g=1)
- `num_kv_heads = num_query_heads / num_groups`

### Multi-head Latent Attention (MLA)
- Compresses KV into low-rank latent representations
- Uses absorption and retrieval operations
- Significantly reduces memory usage for long sequences

Run your implementation with: `python ../../run_assignment.py attention_mechanisms`