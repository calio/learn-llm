# Math: Scaled Dot-Product Attention

Given Q ∈ R^{n_q×d_k}, K ∈ R^{n_k×d_k}, V ∈ R^{n_k×d_v}:

- Scores: S = Q K^T / sqrt(d_k)
- Masked scores: S' = S + M, where M_{ij} ∈ {0, −∞}
- Weights: A = softmax(S')
- Output: O = A V ∈ R^{n_q×d_v}

Causal mask (autoregressive): M_{ij} = −∞ if j > i else 0.

# Self-attention
- Q = X W^Q, K = X W^K, V = X W^V

# Cross-attention
- Q = Y W^Q, K = X W^K, V = X W^V (Y queries X)

# Multi-Head Attention (MHA)
For h heads with W^Q_i, W^K_i, W^V_i, W^O:

- head_i = softmax(Q W^Q_i (K W^K_i)^T / sqrt(d_k) + M) (V W^V_i)
- concat = [head_1; … ; head_h]
- out = concat W^O
