"""Part A runner: correctness + benchmark + bytecode + GIL experiments.

Run:  python setup.py build_ext --inplace  &&  python bench_dot.py
"""
import dis
import threading
import time

import numpy as np


def dot_py(a, b):
    # TODO: pure-Python dot product using a plain for loop (no numpy, no sum()/zip tricks needed).
    raise NotImplementedError("implement dot_py()")


def timeit(fn, *args, reps=5):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def cpu_bound(n=3_000_000):
    s = 0
    for i in range(n):
        s += i * i
    return s


def gil_experiment():
    """Given: runs the same CPU-bound work sequentially and in 2 threads."""
    t0 = time.perf_counter()
    cpu_bound(); cpu_bound()
    seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    ts = [threading.Thread(target=cpu_bound) for _ in range(2)]
    [t.start() for t in ts]; [t.join() for t in ts]
    thr = time.perf_counter() - t0
    print(f"sequential: {seq:.3f}s   2 threads: {thr:.3f}s   speed-up: {seq / thr:.2f}x")


if __name__ == "__main__":
    # --- bytecode (question 1 in TASK.md) ---
    print("== bytecode of dot_py ==")
    dis.dis(dot_py)

    import dot as cy  # compiled Cython module

    n = 1_000_000
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    la, lb = a.tolist(), b.tolist()

    expected = float(np.dot(a, b))
    assert abs(cy.dot(a, b) - expected) < 1e-6 * max(1, abs(expected)), "cython dot is wrong"
    assert abs(dot_py(la, lb) - expected) < 1e-6 * max(1, abs(expected)), "python dot is wrong"
    try:
        cy.dot(a, b[:10]); raise SystemExit("cython dot must raise ValueError on length mismatch")
    except ValueError:
        pass
    print("correctness: OK")

    print("\n== timings (best of 5), n = 1,000,000 ==")
    rows = [("pure python", timeit(dot_py, la, lb)),
            ("cython", timeit(cy.dot, a, b)),
            ("numpy np.dot", timeit(np.dot, a, b))]
    for name, t in rows:
        print(f"{name:14s} {t * 1e3:10.3f} ms   ({rows[0][1] / t:8.1f}x vs python)")

    print("\n== GIL experiment ==")
    gil_experiment()

    print("\n== refcounts (question 3 in TASK.md) ==")
    import sys
    x = []
    print("refs to a fresh list:", sys.getrefcount(x))
    y = x
    print("after y = x:        ", sys.getrefcount(x))
