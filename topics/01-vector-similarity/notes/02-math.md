# Math: Definitions

Let vectors be a = (a₁, …, aₙ) and b = (b₁, …, bₙ).

- Dot product: a · b = ∑ᵢ aᵢ bᵢ
- L2 norm: ||a||₂ = sqrt(∑ᵢ aᵢ²)
- Cosine similarity: cos_sim(a, b) = (a · b) / (||a||₂ ||b||₂)
- Cosine distance: 1 − cos_sim(a, b)
- Euclidean distance (L2): d₂(a, b) = sqrt(∑ᵢ (aᵢ − bᵢ)²)
- Manhattan distance (L1): d₁(a, b) = ∑ᵢ |aᵢ − bᵢ|
- Chebyshev distance (L∞): d∞(a, b) = maxᵢ |aᵢ − bᵢ|
- Minkowski distance (Lp): d_p(a, b) = (∑ᵢ |aᵢ − bᵢ|^p)^(1/p)
- Angular distance: θ(a, b) = arccos(cos_sim(a, b)) ∈ [0, π]

Binary vector set-based metrics:
- Hamming distance: number of positions where aᵢ ≠ bᵢ
- Jaccard similarity (binary): J = |A ∩ B| / |A ∪ B|, where A,B are index sets with 1s

Notes:
- Add small ε to norms to avoid division by zero when using cosine.
- Angular distance is related to cosine: θ = arccos(cos_sim).
