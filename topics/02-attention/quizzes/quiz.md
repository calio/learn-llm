# Quiz: Attention

1) In self-attention, which of the following is true?
   - A. Q, K, V come from different sequences
   - B. Q, K, V all come from the same sequence
   - C. Only Q and K are learned
   - D. Only V is projected

2) Why do we scale QK^T by sqrt(d_k)?
   - A. To reduce computation
   - B. To keep the softmax in a reasonable range and avoid saturation
   - C. To normalize values to [0,1]
   - D. To apply dropout

3) Cross-attention is used typically in:
   - A. Encoder-only models
   - B. Decoder-only models
   - C. Encoder-decoder models
   - D. RNNs only

Answers: 1-B, 2-B, 3-C
