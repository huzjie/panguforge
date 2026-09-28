import unittest
from panguforge.core.moe import MoELayer


class TestMoE(unittest.TestCase):
    def test_forward_shape(self):
        moe = MoELayer(dim=64, hidden_dim=256, num_experts=8, top_k=2, seed=0)
        x = [0.01] * 64
        out, gate = moe(x)
        self.assertEqual(len(out), 64)
        self.assertEqual(len(gate), 8)


if __name__ == "__main__":
    unittest.main()
