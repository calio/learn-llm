# Measuring Attention Sink

Steps:
1. Collect attention weights A^{(l,h)} ∈ R^{T×T} for each layer l and head h on representative prompts.
2. Aggregate across heads/layers (mean or weighted) to get sink profile s_j = mean_{l,h,i} A^{(l,h)}_{i,j}.
3. Compare s_j against token types/positions; look for consistently high mass near BOS/system tokens.
4. Perform ablations (shuffle headers, move instructions) and re-measure.

Pseudo-code (framework-agnostic):

```python
# given a model that returns per-layer, per-head attention weights
weights = model.run_with_attn(prompt_tokens)
# weights: List[L] of arrays [H, T, T]
import numpy as np
stack = np.stack(weights)          # [L, H, T, T]
sink_profile = stack.mean(axis=(0,1))   # [T, T] avg over layers, heads
sink_target = sink_profile.mean(axis=0) # mean attention received per position
```
