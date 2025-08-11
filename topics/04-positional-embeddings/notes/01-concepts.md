# Concepts: Positional Embeddings

Transformers are permutation-invariant over sequence positions. Positional information is added to restore order.

## Sinusoidal (Original Transformer)
- Absolute position p has a deterministic embedding E[p] with sin/cos at geometrically spaced frequencies.
- Properties: extrapolates to unseen lengths, no learned parameters, encodes relative offsets via phase differences.
- Added to token embeddings before attention.

## Learned absolute positions
- A learned table E[p]. Simple and effective for bounded context; does not extrapolate beyond training length.

## Relative position encodings (e.g., T5 bias)
- Add position-dependent biases to attention logits based on relative offset (i−j). Efficient and flexible.

## Rotary Position Embeddings (RoPE)
- Apply a rotation to Q and K in each 2D subspace of the hidden dimension based on position.
- Dot products incorporate relative position by angle subtraction: ⟨R_θi q_i, R_θj k_j⟩ depends on (θi−θj).
- Good extrapolation when using appropriate base frequencies; widely used in modern LLMs.

## 2D/3D positional encodings
- For images (2D): combine separate encodings for x and y axes (additive, concatenation, or 2D RoPE). ViT often uses learned 2D abs positions.
- For 3D (point clouds, volumes): extensions using 3-axis sinusoids or spherical harmonics; or relative biases based on Euclidean offsets.

Trade-offs:
- Absolute (sinusoidal/learned) are simple; relative/rotary often better capture translation invariance and long-range generalization.
