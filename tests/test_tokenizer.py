import unittest
from panguforge.data.tokenizer import MockTokenizer


class TestTokenizer(unittest.TestCase):
    def test_roundtrip(self):
        t = MockTokenizer(vocab_size=512)
        ids = t.encode("hello")
        self.assertEqual(t.decode(ids), "hello")

    def test_vocab_size(self):
        t = MockTokenizer(vocab_size=256)
        self.assertEqual(t.vocab_size, 256)


if __name__ == "__main__":
    unittest.main()
