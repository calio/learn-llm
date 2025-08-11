# Math: Positional Encodings

## Sinusoidal (absolute)
For position p and hidden dimension index 2i:
- E[p, 2i]   = sin(p / 10000^{2i/d})
- E[p, 2i+1] = cos(p / 10000^{2i/d})

Add to token embedding x_p: x'_p = x_p + E[p].

## Rotary Position Embeddings (RoPE)
Partition the d-dimensional vector into d/2 2D planes. For position p, define angle vector Θ_p where Θ_{p,i} = p · α_i with base frequencies α_i.
Apply rotation per 2D pair (x_{2i}, x_{2i+1}):

- R(Θ_{p,i}) [x_{2i}, x_{2i+1}] = [x_{2i} cos Θ_{p,i} − x_{2i+1} sin Θ_{p,i},
                                   x_{2i} sin Θ_{p,i} + x_{2i+1} cos Θ_{p,i}]

Then use Q' = RoPE(Q), K' = RoPE(K) in attention. Inner products become functions of relative angle differences Θ_{p} − Θ_{q}.

## 2D/3D extensions
- 2D: use separate sin/cos for x and y and add/concat; or apply RoPE with angles derived from (px, py).
- 3D: extend with (px, py, pz) or spherical bases; for relative, use offsets (Δx, Δy, Δz) to parameterize biases or rotations.
