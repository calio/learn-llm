# Attention Sink

Attention sink refers to specific positions (often early context tokens or special tokens) that systematically attract attention mass from later tokens. This can act as a "sink" for attention, influencing model behavior (e.g., focusing on prepended prompts, control tokens, or positional anchors) in generation and retrieval-augmented settings.

In this unit:
- Define attention sink phenomenon and its empirical observations
- Show how causal masks and positional embeddings interplay with sinks
- Explore mitigation and usage strategies (e.g., prompt engineering, controlled headers)
- Provide code to measure and visualize sink patterns
