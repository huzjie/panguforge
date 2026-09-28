"""KV-cache for incremental decoding."""


class KVCache:
    def __init__(self, max_len=1024):
        self.keys = []
        self.values = []
        self.max_len = max_len

    def push(self, k, v):
        self.keys.append(k)
        self.values.append(v)
        if len(self.keys) > self.max_len:
            self.keys.pop(0)
            self.values.pop(0)

    def clear(self):
        self.keys.clear()
        self.values.clear()

    def __len__(self):
        return len(self.keys)
