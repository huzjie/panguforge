import unittest
from panguforge.config import load_config
from panguforge.backends import get_backend
from panguforge.data import SyntheticDataset


class TestMockTraining(unittest.TestCase):
    def test_skill_increases(self):
        cfg = load_config("examples/config_pretrain.yaml")
        eng = get_backend("mock", cfg)
        ds = SyntheticDataset(vocab_size=512, seed=0)
        before = eng.skill
        for _ in range(50):
            eng.train_step(ds.sample_batch())
        self.assertGreater(eng.skill, before)

    def test_loss_decreases(self):
        cfg = load_config("examples/config_pretrain.yaml")
        eng = get_backend("mock", cfg)
        ds = SyntheticDataset(vocab_size=512, seed=0)
        l0 = eng.compute_loss(ds.sample_batch())
        for _ in range(50):
            eng.train_step(ds.sample_batch())
        l1 = eng.compute_loss(ds.sample_batch())
        self.assertLess(l1, l0)


if __name__ == "__main__":
    unittest.main()
