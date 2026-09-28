"""Minimal zero-dependency tensor library (pure Python, small dims only).

Real frameworks offload this to torch/numpy; panguforge keeps a tiny, correct
subset so the CPU backend can run genuine forward passes with no third-party
dependency. Shapes are small (hidden ~64) so nested-list performance is fine.
"""
import math
import random


def shape(x):
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
        return [len(x), len(x[0])]
    if isinstance(x, (list, tuple)):
        return [len(x)]
    return []


def zeros(m, n=None):
    if n is None:
        return [0.0] * m
    return [[0.0] * n for _ in range(m)]


def ones(m, n=None):
    if n is None:
        return [1.0] * m
    return [[1.0] * n for _ in range(m)]


def randn(m, n=None, seed=None, scale=0.02):
    rng = random.Random(seed) if seed is not None else random
    if n is None:
        return [rng.gauss(0, scale) for _ in range(m)]
    return [[rng.gauss(0, scale) for _ in range(n)] for _ in range(m)]


def matmul(A, B):
    """A: [m,k], B: [k,n] -> [m,n]."""
    m = len(A)
    k = len(B)
    n = len(B[0])
    Bt = [[B[i][j] for i in range(k)] for j in range(n)]
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        ai = A[i]
        oi = out[i]
        for j in range(n):
            bj = Bt[j]
            s = 0.0
            for t in range(k):
                s += ai[t] * bj[t]
            oi[j] = s
    return out


def matvec(A, x):
    """A: [m,k], x: [k] -> [m]."""
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def add(A, B):
    return [a + b for a, b in zip(A, B)]


def add_scalar(A, s):
    if isinstance(A[0], (list, tuple)):
        return [[a + s for a in row] for row in A]
    return [a + s for a in A]


def mul_scalar(A, s):
    if isinstance(A[0], (list, tuple)):
        return [[a * s for a in row] for row in A]
    return [a * s for a in A]


def transpose(A):
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def softmax(x):
    mx = max(x)
    ex = [math.exp(v - mx) for v in x]
    s = sum(ex)
    return [e / s for e in ex]


def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])


def relu(x):
    if isinstance(x[0], (list, tuple)):
        return [[max(0.0, v) for v in row] for row in x]
    return [max(0.0, v) for v in x]


def gelu(x):
    c = 0.7978845608028654  # sqrt(2/pi)
    if isinstance(x[0], (list, tuple)):
        return [[0.5 * v * (1.0 + math.tanh(c * (v + 0.044715 * v ** 3))) for v in row] for row in x]
    return [0.5 * v * (1.0 + math.tanh(c * (v + 0.044715 * v ** 3))) for v in x]


def silu(x):
    if isinstance(x[0], (list, tuple)):
        return [[v / (1.0 + math.exp(-v)) for v in row] for row in x]
    return [v / (1.0 + math.exp(-v)) for v in x]


def rms_norm(x, weight=None, eps=1e-6):
    n = len(x)
    ms = sum(v * v for v in x) / n
    inv = 1.0 / math.sqrt(ms + eps)
    if weight is not None:
        return [x[i] * inv * weight[i] for i in range(n)]
    return [v * inv for v in x]


def layer_norm(x, weight=None, bias=None, eps=1e-5):
    n = len(x)
    mu = sum(x) / n
    var = sum((v - mu) ** 2 for v in x) / n
    inv = 1.0 / math.sqrt(var + eps)
    out = []
    for i in range(n):
        v = (x[i] - mu) * inv
        if weight is not None:
            v *= weight[i]
        if bias is not None:
            v += bias[i]
        out.append(v)
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class TensorOps:
    """Namespace re-exporting all primitives for ergonomic import."""

    shape = staticmethod(shape)
    zeros = staticmethod(zeros)
    ones = staticmethod(ones)
    randn = staticmethod(randn)
    matmul = staticmethod(matmul)
    matvec = staticmethod(matvec)
    add = staticmethod(add)
    mul_scalar = staticmethod(mul_scalar)
    transpose = staticmethod(transpose)
    softmax = staticmethod(softmax)
    argmax = staticmethod(argmax)
    relu = staticmethod(relu)
    gelu = staticmethod(gelu)
    silu = staticmethod(silu)
    rms_norm = staticmethod(rms_norm)
    layer_norm = staticmethod(layer_norm)
    dot = staticmethod(dot)
