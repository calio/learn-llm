# Implementation Notes

- Use float32 or bfloat16; apply attention-scale 1/sqrt(d_k) before softmax.
- Apply masks by adding a large negative number to disallowed logits (e.g., -1e9) before softmax.
- Numerical stability: subtract row-wise max from logits before softmax.
- Efficient batching: use batched Q,K,V tensors of shape [B, T, D].
- Causal mask: lower triangular mask for autoregressive decoders.
- Cross-attention: separate K,V from encoder; optionally cache K,V for decoding.
- MQA/GQA: share K/V across heads to reduce memory bandwidth while keeping multiple query heads.
