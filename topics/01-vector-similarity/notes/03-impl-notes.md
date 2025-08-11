# Implementation Notes

- Numerical stability: add a small ε (e.g., 1e-12) to norms before division.
- Normalization: Pre-normalize vectors (unit length) to turn cosine similarity into inner product.
- Performance: Batch computations using matrix ops (NumPy/PyTorch) for throughput.
- High dimensionality: Distances can concentrate; choose thresholds via empirical calibration.
- Index compatibility: HNSW/FAISS often optimize for L2 or inner-product; map cosine to inner-product by normalizing.
- Binary metrics: Ensure inputs are 0/1 when using Jaccard/Hamming variants intended for binary vectors.
