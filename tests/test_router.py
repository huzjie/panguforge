import unittest
from panguforge.core.router import TopKRouter


class TestRouter(unittest.TestCase):
    def test_topk(self):
        r = TopKRouter(dim=64, num_experts=8, top_k=2, seed=0)
        ids, weights, logits = r.route([0.01] * 64)
        self.assertEqual(len(ids), 2)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_load_balance(self):
        r = TopKRouter(dim=64, num_experts=8, top_k=2, seed=0)
        loss = r.load_balancing_loss([[float(i % 8) for i in range(8)] for _ in range(16)])
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
