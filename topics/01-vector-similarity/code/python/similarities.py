"""Vector similarity and distance metrics (pure Python).

Intended for learning and small demos. For production-scale workloads, prefer
vectorized libraries such as NumPy, PyTorch, or specialized ANN libraries.

Functions:
- dot_product
- l2_norm
- cosine_similarity / cosine_distance
- euclidean_distance (L2)
- manhattan_distance (L1)
- chebyshev_distance (L_inf)
- minkowski_distance (L_p)
- angular_distance
- hamming_distance
- jaccard_similarity_binary

All functions validate input lengths and handle simple edge cases.
"""
from __future__ import annotations

from math import acos, isclose, pow, sqrt
from typing import Sequence

Number = float
Vector = Sequence[Number]

_EPS = 1e-12


def _ensure_same_length(a: Vector, b: Vector) -> None:
    if len(a) != len(b):
        raise ValueError(f"Vectors must have same length: {len(a)} != {len(b)}")


def dot_product(a: Vector, b: Vector) -> Number:
    """Compute the dot product of two vectors.

    >>> dot_product([1, 2], [3, 4])
    11.0
    """
    _ensure_same_length(a, b)
    return float(sum(x * y for x, y in zip(a, b)))


def l2_norm(a: Vector) -> Number:
    """Compute L2 norm of a vector.

    >>> isclose(l2_norm([3, 4]), 5.0)
    True
    """
    return sqrt(sum(x * x for x in a))


def cosine_similarity(a: Vector, b: Vector) -> Number:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is near zero.

    >>> round(cosine_similarity([1, 0], [0, 1]), 6)
    0.0
    >>> isclose(cosine_similarity([1, 0], [1, 0]), 1.0)
    True
    """
    _ensure_same_length(a, b)
    na = l2_norm(a)
    nb = l2_norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return dot_product(a, b) / (na * nb)


def cosine_distance(a: Vector, b: Vector) -> Number:
    """Cosine distance = 1 - cosine_similarity.

    >>> isclose(cosine_distance([1, 0], [1, 0]), 0.0)
    True
    """
    return 1.0 - cosine_similarity(a, b)


def euclidean_distance(a: Vector, b: Vector) -> Number:
    """Euclidean (L2) distance.

    >>> isclose(euclidean_distance([0, 0], [3, 4]), 5.0)
    True
    """
    _ensure_same_length(a, b)
    return sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


def manhattan_distance(a: Vector, b: Vector) -> Number:
    """Manhattan (L1) distance.

    >>> isclose(manhattan_distance([1, 2], [3, 4]), 4.0)
    True
    """
    _ensure_same_length(a, b)
    return float(sum(abs(x - y) for x, y in zip(a, b)))


def chebyshev_distance(a: Vector, b: Vector) -> Number:
    """Chebyshev (L_inf) distance.

    >>> isclose(chebyshev_distance([1, 5, 2], [3, 4, 8]), 6.0)
    True
    """
    _ensure_same_length(a, b)
    return float(max(abs(x - y) for x, y in zip(a, b)))


def minkowski_distance(a: Vector, b: Vector, p: Number) -> Number:
    """Minkowski Lp distance for p >= 1.

    >>> isclose(minkowski_distance([0, 0], [3, 4], p=2), 5.0)
    True
    """
    if p < 1:
        raise ValueError("p must be >= 1 for a valid metric")
    _ensure_same_length(a, b)
    return pow(sum(pow(abs(x - y), p) for x, y in zip(a, b)), 1.0 / p)


def angular_distance(a: Vector, b: Vector) -> Number:
    """Angular distance in radians, in [0, pi].

    >>> isclose(round(angular_distance([1, 0], [0, 1]), 6), 1.570796)
    True
    """
    c = max(-1.0, min(1.0, cosine_similarity(a, b)))
    return float(acos(c))


def hamming_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Hamming distance: count of positions that differ. Requires equal length.

    Typically used for binary vectors, but works for any discrete symbols.

    >>> hamming_distance([1, 0, 1, 1], [1, 1, 0, 1])
    2
    """
    _ensure_same_length(a, b)
    return sum(1 for x, y in zip(a, b) if x != y)


def jaccard_similarity_binary(a: Sequence[int], b: Sequence[int]) -> Number:
    """Jaccard similarity for binary vectors (0/1).

    J = |A ∩ B| / |A ∪ B| where A,B are sets of indices with value 1.

    >>> isclose(jaccard_similarity_binary([1, 1, 0], [1, 0, 1]), 1/3)
    True
    """
    _ensure_same_length(a, b)
    intersection = 0
    union = 0
    for x, y in zip(a, b):
        if x == 1 or y == 1:
            union += 1
        if x == 1 and y == 1:
            intersection += 1
    if union == 0:
        return 1.0  # both empty
    return intersection / union


if __name__ == "__main__":
    a = [1.0, 0.0]
    b = [0.7, 0.7]
    print("a:", a)
    print("b:", b)
    print("cosine_similarity:", cosine_similarity(a, b))
    print("cosine_distance:", cosine_distance(a, b))
    print("euclidean_distance:", euclidean_distance(a, b))
    print("manhattan_distance:", manhattan_distance(a, b))
    print("chebyshev_distance:", chebyshev_distance(a, b))
    print("angular_distance:", angular_distance(a, b))
