"""Benchmark suite: language modeling, long-context, throughput."""
import time

from ..backends import get_backend
from ..data import SyntheticDataset
from ..train.eval import perplexity, accuracy


def bench_language_modeling(engine, ds):
    before = perplexity(engine, ds, n=50)
    # train a few steps
    for _ in range(30):
        engine.train_step(ds.sample_batch())
    after = perplexity(engine, ds, n=50)
    acc = accuracy(engine, ds, n=200)
    return {"perplexity_before": before, "perplexity_after": after, "accuracy": acc}


def bench_throughput(engine):
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        engine.generate("hello world", max_tokens=8)
    dt = time.perf_counter() - t0
    return {"requests": n, "seconds": round(dt, 3), "req_per_sec": round(n / max(dt, 1e-6), 2)}


def bench_long_context(engine):
    # simulate long-context scaling: loss of a 4k-token synthetic sequence
    seq = list(range(1, 4096))
    t0 = time.perf_counter()
    loss = engine.compute_loss([{"input": seq}])
    dt = time.perf_counter() - t0
    return {"context_tokens": len(seq), "loss": round(loss, 4), "seconds": round(dt, 4)}


def run_bench(config):
    eng = get_backend(config.get("backend", "mock"), config)
    ds = SyntheticDataset(vocab_size=int(config.get("model", {}).get("vocab_size", 512)), seed=0)
    out = {}
    out["language_modeling"] = bench_language_modeling(eng, ds)
    out["throughput"] = bench_throughput(eng)
    out["long_context"] = bench_long_context(eng)
    return out
