import unittest
from panguforge.core.attention import MultiHeadAttention
from panguforge.core.attention_sparse import SlidingWindowAttention
from panguforge.core.attention_dsa import SparseAttentionDSA


class TestAttention(unittest.TestCase):
    def test_full_attention(self):
        attn = MultiHeadAttention(dim=64, num_heads=4, seed=0)
        x = [[0.01] * 64 for _ in range(6)]
        out = attn(x)
        self.assertEqual(len(out), 6)
        self.assertEqual(len(out[0]), 64)

    def test_swa(self):
        attn = SlidingWindowAttention(dim=64, num_heads=4, window=3, seed=0)
        out = attn([[0.01] * 64 for _ in range(6)])
        self.assertEqual(len(out), 6)

    def test_dsa(self):
        attn = SparseAttentionDSA(dim=64, num_heads=4, n_select=2, seed=0)
        out = attn([[0.01] * 64 for _ in range(6)])
        self.assertEqual(len(out), 6)


if __name__ == "__main__":
    unittest.main()
