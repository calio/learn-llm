# YaRN: Yet Another RoPE Extension for Long Context

Goal: extend a RoPE-trained model (context length N_train) to a longer target length L_target without retraining, while preserving behavior on short ranges.

## Why RoPE degrades at long positions
- RoPE rotates Q/K by position-dependent angles. High-frequency components rotate very fast at large positions.
- Extrapolating far beyond training positions causes phase aliasing and attention collapse (very peaky or flat weights).

## Existing fixes (recap)
- Linear scaling: use effective position p' = p / s (s = L_target / N_train). Simple but over-compresses short-range angles; can regress near the beginning.
- NTK-aware scaling (a.k.a. dynamic interpolation): compress angles nonlinearly so low frequencies are preserved more than high; better near the training window.

## YaRN idea (high level)
- Piecewise/continuous remapping of positions p → f(p) that:
  - Keeps f(p) ≈ p for p ≤ N_train (preserve in-distribution behavior)
  - Gradually increases compression beyond N_train using a smooth "ramp"
  - Uses different compression for low vs. high frequencies so relative information is retained
- Result: better stability across the whole range, stronger retention at short ranges, and usable attention at long ranges.

## Practical recipe (conceptual)
Choose:
- N_train: original max context (e.g., 4k)
- L_target: desired context (e.g., 128k)
- s = L_target / N_train: overall scale
- ramp ∈ [0,1]: fraction of N_train used to smoothly start compressing beyond N_train
- Two scales: s_low ≈ √s (gentle), s_high ≈ s (strong)

Define f(p):
- If p ≤ N_train: f(p) = p
- Let extra = p − N_train and R = ramp · N_train
- If 0 < extra ≤ R: f(p) = N_train + extra / s_low
- If extra > R: f(p) = N_train + R / s_low + (extra − R) / s_high

Then apply RoPE with angles θ_i(p) = f(p) · α_i, where α_i are base frequencies per 2D plane (same as standard RoPE).

This schedule keeps early positions nearly unchanged, compresses moderately just after N_train, and fully compresses as p grows, protecting short-range behavior while allowing much longer contexts.

## Pseudo-code
```python
def yarn_position_map(p: int, n_train: int, scale: float, ramp_frac: float = 0.5):
    s_low = max(1.0, scale ** 0.5)
    s_high = max(1.0, scale)
    if p <= n_train:
        return float(p)
    extra = p - n_train
    R = int(n_train * ramp_frac)
    if extra <= R:
        return n_train + extra / s_low
    return n_train + R / s_low + (extra - R) / s_high

# Use in RoPE
# angle_i(p) = yarn_position_map(p, N_train, L_target/N_train, ramp_frac) * alpha_i
```

Notes:
- Exact functions in public YaRN variants may differ; the key is smooth, frequency-aware compression with identity near the training span.
- Choose ramp_frac and s_low to meet quality targets on short contexts while enabling the desired long range.
- Combine with serving optimizations (GQA/MQA, PagedAttention) for throughput.

## When to use YaRN
- Extending LLaMA-style models from 4k/8k to 32k–128k+ without full retraining.
- Maintaining task performance on short prompts while enabling longer documents.
