# Sprint 1 Task: A C-speed dot product and a strided tensor

*Days 1–2 · about 4 hours · read `ARTICLE.md` first (20 minutes)*

You will build two things. Part A shows how much a typed loop in C beats a Python loop. Part B builds the storage-and-strides core of a tensor in C++.

## Files (in `task/`)

| File | You do |
|---|---|
| `dot.pyx` | Implement `dot()` (Cython) |
| `bench_dot.py` | Implement `dot_py()`; run the benchmarks |
| `setup.py` | Given: builds the Cython module |
| `tensor.hpp` | Implement the `TODO`s (C++) |
| `test_tensor.cpp` | Given: the tests for Part B |

## Setup

```bash
pip install numpy cython setuptools
# macOS: xcode-select --install     Ubuntu: sudo apt install build-essential
```

## Part A: dot product (about 1.5 h)

1. Implement `dot_py(a, b)` in `bench_dot.py` with a plain `for` loop.
2. Implement `dot(a, b)` in `dot.pyx` using typed memoryviews (`double[::1]`), `cdef` loop variables, and a `ValueError` when the lengths differ.
3. Build and run:

```bash
python setup.py build_ext --inplace
python bench_dot.py
```

**Pass condition:** `correctness: OK` prints, and Cython is at least 20x faster than pure Python at n = 1,000,000.

**Answer these in a short `NOTES.md` (2–4 sentences each):**

1. In the `dis` output of `dot_py`, which instructions are executed once per loop iteration? Which ones do the boxing and dynamic dispatch?
2. Why is Cython faster than the Python loop? Why is `np.dot` usually still a bit faster than your Cython loop?
3. What did the GIL experiment show, and why? What would change if the work were a NumPy matrix multiply, or waiting on a network call?
4. The refcount output shows `2` for a fresh list. Why not `1`?

**Stretch:** run with `boundscheck=True, wraparound=True` and measure the cost of the safety checks.

## Part B: strided tensor in C++ (about 2.5 h)

Implement everything marked `TODO` in `tensor.hpp`: `Storage`, the `Tensor` constructor (row-major strides), `at`, `is_contiguous`, `transpose`, `select`, `contiguous`, `reshape`. Each stub throws `NotImplemented`; delete that line as you implement.

```bash
g++ -std=c++17 -O1 -Wall -Wextra test_tensor.cpp -o test_tensor && ./test_tensor
```

**Pass condition:** `10/10 passed`, with no compiler warnings.

**Rules**
- `Storage` owns memory through `std::shared_ptr<float[]>`. No raw `new`/`delete` outside it.
- `transpose`, `select`, and `reshape` must be **views** (the tests compare storage pointers).
- `contiguous()` must return **new storage**.

**Answer in `NOTES.md`:**
5. After `t.transpose(0, 1)` on a `(3, 4)` tensor, what are the shape and strides? Is it contiguous? Why can't you `reshape` it directly?
6. What would break if `Storage` held a raw `float*` and freed it in its destructor, while views copied that pointer? Which test would catch it?

**Stretch:** add `permute(std::vector<int64_t> dims)`, and `fill(float)`/`arange(n)` helpers so the tests stop using `const_cast`.

## Done when

- [ ] `bench_dot.py` correctness passes and the timing table is in `NOTES.md`
- [ ] `10/10 passed` for the C++ tests
- [ ] The six answers are in `NOTES.md`
- [ ] Committed as `Day 1-2: cython dot + strided tensor`

## If you are short on time

Do Part A steps 1–3 and skip the questions on the GIL and refcounts. In Part B, implement only the constructor, `at`, `transpose` and `contiguous`; the rest can wait.

*Reference solutions are in `solutions/sprint1/`. Try first, then compare.*
