# Concepts: Attention

Attention lets a model compute context-aware representations by weighting other tokens (or encoder states) based on learned relevance. Queries (Q) ask “what am I looking for?”, keys (K) say “what do I contain?”, and values (V) are the content to aggregate.

## Self-attention
- Q, K, V all come from the same sequence X (e.g., a token sequence in a transformer block).
- Each position i attends to all positions j in the same sequence, producing a contextualized representation for i.
- Bidirectional self-attention (BERT-style): positions attend both left and right; great for encoding.
- Unidirectional (causal) self-attention (GPT-style): each position i attends only to j ≤ i to preserve autoregressive factorization.
- Use-cases: language modeling, contextual embeddings, encoder blocks.

## Cross-attention
- Q comes from a target/decoder sequence Y; K and V come from a source/encoder sequence X.
- Lets a decoder query relevant information from the encoded inputs (seq2seq). Common in translation, summarization, speech, and vision-language models.
- In encoder–decoder transformers (e.g., T5), a decoder block typically has: self-attention on Y (causal), then cross-attention over encoder outputs X, then feed-forward.

## Causal attention (masked self-attention)
- Enforces autoregressive constraint by masking out future tokens with a lower-triangular mask (−∞ added to logits before softmax).
- Enables next-token prediction without information leakage from the future.
- At inference time, key/value caching makes causal attention O(T) per new token (vs naive O(T^2)).

## Multi-head attention (MHA)
- Multiple parallel attention heads attend to different subspaces/relations; outputs are concatenated and linearly projected.
- Practical variants: MQA (shared K/V per layer), GQA (grouped K/V among heads) to reduce memory bandwidth with minimal quality loss.

## Practical notes
- Scaling by 1/sqrt(d_k) stabilizes softmax.
- Masks: padding masks (ignore pad positions), causal masks, and task-specific masks.
- Positional information: absolute (sinusoidal/learned), relative biases, or rotary embeddings (RoPE).
- Complexity: standard attention is O(T^2 d). Long-context models use sparse/windowed/linear attention or memory mechanisms.
