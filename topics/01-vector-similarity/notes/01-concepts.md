# Concepts: Vector Similarity

Vector similarity (or distance) quantifies how "close" two vectors are. In embedding-based systems (semantic search, RAG, clustering), we choose a metric and an index structure to compare vectors efficiently.

Common use-cases:
- Nearest Neighbor search for retrieval
- Deduplication and clustering
- Reranking candidates by semantic proximity

Similarity vs distance:
- Similarity increases with closeness (e.g., cosine similarity in [−1, 1])
- Distance decreases with closeness (e.g., Euclidean distance in [0, ∞))

Key considerations:
- Normalization: Cosine uses direction only; Euclidean is sensitive to magnitude.
- Scale: Feature scaling impacts Lp distances; cosine mitigates scale effects.
- Dimensionality: Distances can concentrate in high-D; calibrate thresholds empirically.
- Indexing: ANN libraries (FAISS, ScaNN, HNSW) often support inner-product or L2; cosine can be mapped to inner-product by normalizing vectors.
