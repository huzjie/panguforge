import unittest
from panguforge.config import load_config


class TestConfig(unittest.TestCase):
    def test_load_yamlish(self):
        cfg = load_config("examples/config_pretrain.yaml")
        self.assertEqual(cfg["backend"], "mock")
        self.assertEqual(cfg["model"]["vocab_size"], 512)

    def test_getattr(self):
        cfg = load_config("examples/config_pretrain.yaml")
        self.assertEqual(cfg.get("backend"), "mock")


if __name__ == "__main__":
    unittest.main()
