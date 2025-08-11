from __future__ import annotations

from similarities import (
    angular_distance,
    chebyshev_distance,
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
)


def main() -> None:
    a = [1.0, 0.0]
    b = [0.7, 0.7]
    print("Vector A:", a)
    print("Vector B:", b)
    print("- cosine_similarity:", round(cosine_similarity(a, b), 6))
    print("- cosine_distance:", round(cosine_distance(a, b), 6))
    print("- euclidean_distance:", round(euclidean_distance(a, b), 6))
    print("- manhattan_distance:", round(manhattan_distance(a, b), 6))
    print("- chebyshev_distance:", round(chebyshev_distance(a, b), 6))
    print("- angular_distance (rad):", round(angular_distance(a, b), 6))


if __name__ == "__main__":
    main()
