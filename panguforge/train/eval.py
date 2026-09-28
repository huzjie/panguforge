"""Evaluation helpers."""


def accuracy(engine, dataset, n=100):
    correct = 0
    wt = getattr(engine, "world_target", None)
    for _ in range(n):
        ex = dataset.sample_example()
        inp = ex["input"]
        pred = engine.next_token(inp)
        gold = wt(str(list(inp))) if wt is not None else ex["target"]
        if pred == gold:
            correct += 1
    return correct / n


def perplexity(engine, dataset, n=100):
    import math
    total = 0.0
    for _ in range(n):
        ex = dataset.sample_example()
        total += engine.compute_loss([ex])
    return math.exp(total / n)
