"""Simple batch iterator."""
import random


class DataLoader:
    def __init__(self, dataset, batch_size=16, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self):
        n = getattr(self.dataset, "num_examples", 64)
        idx = list(range(n))
        if self.shuffle:
            self.rng.shuffle(idx)
        for i in range(0, n, self.batch_size):
            yield [self.dataset.sample_example() for _ in idx[i:i + self.batch_size]]
