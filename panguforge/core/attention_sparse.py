"""Sliding-Window Attention (SWA) — local window mask, O(w) per token."""
from .attention import MultiHeadAttention


def sliding_window_mask(window):
    def mask(i, j):
        return j <= i and (i - j) < window
    return mask


class SlidingWindowAttention(MultiHeadAttention):
    def __init__(self, dim, num_heads, window, seed=0):
        super().__init__(dim, num_heads, seed)
        self.window = window

    def __call__(self, x):
        return super().__call__(x, mask=sliding_window_mask(self.window))
