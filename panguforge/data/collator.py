"""Batch collation (left-pad to equal length)."""


def pad_batch(examples, pad_id=0):
    if not examples:
        return []
    max_len = max(len(e["input"]) for e in examples)
    inputs, targets = [], []
    for e in examples:
        inp = e["input"]
        inputs.append([pad_id] * (max_len - len(inp)) + list(inp))
        targets.append(e["target"])
    return {"input": inputs, "target": targets}
