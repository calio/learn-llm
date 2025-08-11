# Vector Similarity

This unit introduces vector similarity and distance metrics used throughout information retrieval, embeddings, and LLM applications (RAG, clustering, semantic search).

Contents:
- Concepts and math notes under `notes/`
- Python implementations under `code/python/`
- Interactive 2D visualization under `visuals/interactive/vector-similarity-explorer/`
- Tiny example vectors under `data/tiny/`
- Quiz prompts under `quizzes/`
- References under `references/`

Key metrics covered:
- Cosine similarity and cosine distance
- Dot product
- Euclidean (L2), Manhattan (L1), Chebyshev (L∞), Minkowski (Lp)
- Angular distance
- Hamming and Jaccard (for binary vectors)

How to use the visualization:
1. Open `visuals/interactive/vector-similarity-explorer/index.html` in a browser.
2. Adjust the components of vectors A and B.
3. Switch metrics to see how values change.

How to run the Python demo:
```bash
python3 topics/01-vector-similarity/code/python/demo.py
```
