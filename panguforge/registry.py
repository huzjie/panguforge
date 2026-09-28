"""Generic named registry."""


class Registry:
    def __init__(self, name):
        self.name = name
        self._items = {}

    def register(self, key, obj=None):
        def deco(o):
            self._items[key] = o
            return o
        if obj is not None:
            self._items[key] = obj
            return obj
        return deco

    def get(self, key, default=None):
        return self._items.get(key, default)

    def keys(self):
        return list(self._items.keys())

    def __contains__(self, key):
        return key in self._items
