# Fast Self-Attention Mechanisms

This note summarizes engineering variants that improve speed and memory efficiency of attention.

## Multi-Query Attention (MQA)
- Idea: Multiple query heads but share K/V across all heads.
- Benefits: Reduces K/V size and bandwidth (fewer K/V projections and caches), faster decoding.
- Trade-off: Slight loss in expressivity vs full MHA; often minimal quality loss in practice.

## Grouped-Query Attention (GQA)
- Idea: Group heads into g groups; each group shares K/V. MQA is the special case g=1.
- Benefits: Balance between MHA quality and MQA efficiency; reduces memory/computation.
- Use-cases: Large decoder models for serving; better latency–quality trade-off.

## Sliding Window Attention (SWA)
- Idea: Each token attends only to a fixed window of previous tokens (or a pattern combining local + global tokens).
- Benefits: O(T · w) complexity vs O(T^2), where w is window size.
- Trade-off: May miss long-range interactions; augment with periodic global tokens or dilations.

## FlashAttention
- Idea: IO-aware attention kernels that keep tiles in fast on-chip SRAM, compute softmax in a numerically stable tiled manner, and avoid materializing the full attention matrix.
- Benefits: Large speedups and memory savings; exact attention (not approximation) within block-wise computation.
- Notes: Implemented in fused CUDA kernels; widely adopted in modern frameworks.

## PagedAttention
- Idea: Manage K/V caches with a paging scheme to minimize memory fragmentation and enable efficient batching of variable-length sequences (common in serving).
- Benefits: Better GPU memory utilization and throughput under dynamic batching loads.
- Notes: Used in vLLM; critical for high-throughput multi-request decoding.

## When to use what
- Training big models: FlashAttention for kernel efficiency; consider GQA for compute/memory balance.
- Serving long prompts with many concurrent requests: GQA or MQA to shrink K/V cache, plus PagedAttention for batching.
- Long-context tasks with locality: SWA or hybrid patterns (local + sparse globals).

## Further reading
- "The Race for Faster Transformers: Innovations in Self-Attention" — [Medium article](https://medium.com/@lmpo/the-race-for-faster-transformers-innovations-in-self-attention-e602fb1b5f20)
