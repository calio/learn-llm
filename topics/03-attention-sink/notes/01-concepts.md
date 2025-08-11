# Concepts: Attention Sink

An attention sink is a position that absorbs a disproportionate amount of attention across layers/time, often independent of semantic content.

Observations and hypotheses:
- Early tokens (e.g., BOS, system prompt headers) can attract attention mass due to positional encodings and training dynamics.
- Sinks act as anchors that stabilize representations and provide a default fallback when uncertainty is high.
- In causal models, the leftmost context accumulates information; sinks can emerge as robust references for value mixing.

Implications:
- Prompt design: placing key instructions at or near sink positions can improve adherence.
- Retrieval: prepending summaries/headers may increase their influence via sink behavior.
- Evaluation: analyze attention distributions to detect over-reliance on sinks.

Mitigations/uses:
- Reweight attention (research); adjust positional strategies; diversify prompts.
- Intentionally exploit sinks to steer behavior via structured headers.
