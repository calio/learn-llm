from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def scaled_dot_product_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    d_k = q.shape[-1]
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask  # mask should contain 0 or large negative values
    weights = softmax(scores, axis=-1)
    return weights @ v


def causal_mask(t: int) -> np.ndarray:
    # returns shape [1, 1, t, t] suitable for broadcasting
    m = np.triu(np.ones((t, t), dtype=np.float32), k=1)
    m[m == 1] = -1e9
    m[m == 0] = 0
    return m[None, None, :, :]


def multi_head_attention(x: np.ndarray, num_heads: int, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray, wo: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    # x: [B, T, D]; weights shapes: [D, D]
    b, t, d = x.shape
    assert d % num_heads == 0
    head_dim = d // num_heads

    q = x @ wq  # [B, T, D]
    k = x @ wk
    v = x @ wv

    # reshape to heads
    def split_heads(z):
        z = z.reshape(b, t, num_heads, head_dim)
        return np.transpose(z, (0, 2, 1, 3))  # [B, H, T, Hd]

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)

    # attention per head
    out_h = scaled_dot_product_attention(qh, kh, vh, mask=mask)  # [B, H, T, Hd]

    # merge heads
    out = np.transpose(out_h, (0, 2, 1, 3)).reshape(b, t, d)
    out = out @ wo  # [B, T, D]
    return out


if __name__ == "__main__":
    np.random.seed(0)
    B, T, D, H = 2, 4, 8, 2
    x = np.random.randn(B, T, D).astype(np.float32)
    wq = np.random.randn(D, D).astype(np.float32)
    wk = np.random.randn(D, D).astype(np.float32)
    wv = np.random.randn(D, D).astype(np.float32)
    wo = np.random.randn(D, D).astype(np.float32)

    y = multi_head_attention(x, H, wq, wk, wv, wo, mask=causal_mask(T))
    print(y.shape)
