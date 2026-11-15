# From Python Bytecode to Tensor Memory
....Continued
*Sprint 1 reference article · 100 Days of ML, Advanced track, Days 1–2*

Every deep-learning framework is, at the bottom, two ideas stacked together: a slow, flexible language driving fast, rigid memory. This article covers both halves. 
1. First, what CPython actually does with your code, and why it costs so much per operation. 
2. Second, how a tensor is really just a flat block of memory with a small amount of bookkeeping on top. If you understand both, the rest of the Advanced track (autograd, C++ extensions, CUDA) stops feeling like magic.

## Part 1. What CPython does when you run a line of code

### Source, bytecode, and the evaluation loop

CPython does not interpret your source text directly. It compiles each function to **bytecode**, a compact list of instructions for a stack machine, and then a big C loop (the *evaluation loop*, in `ceval.c`) executes those instructions one at a time. You can see the bytecode with the `dis` module:

```python
import dis
def f(a, b):
    return a * b + 1
dis.dis(f)
```

You will see instructions like `LOAD_FAST a`, `LOAD_FAST b`, `BINARY_OP *`, `LOAD_CONST 1`, `BINARY_OP +`, `RETURN_VALUE`. Exact names vary by Python version (3.11+ specialises hot instructions), but the shape is stable: push operands, apply an operation, return.

### Why a Python loop is slow

Consider `total += a[i] * b[i]`. For *each* element CPython must:

1. Fetch `a[i]` and `b[i]` through the sequence protocol, which returns **boxed** `float` objects (a heap-allocated struct with a refcount, a type pointer and an 8-byte payload).
2. Look up the type of each operand and dispatch `*` dynamically (`float.__mul__`, which could in principle be anything).
3. Allocate a **new** float object for the product, then another for the running sum.
4. Update reference counts on every object touched.

That is dozens of machine instructions and several memory touches for one multiply-add that a CPU can do in a fraction of a nanosecond. The cost is not the bytecode count; it is the *boxing and dynamic dispatch per element*. This is why vectorised libraries exist: they move the loop out of Python and into C, where the data is a raw array of unboxed numbers.

### Reference counting and the garbage collector

Every Python object carries a **reference count**. Assigning `y = x` increments it; a name going out of scope decrements it; at zero the object is freed immediately. `sys.getrefcount(x)` reports it (always one higher than you expect, because the function argument itself is a reference). Reference counting cannot free **cycles** (two objects pointing at each other), so CPython also runs a generational cycle collector, controllable through the `gc` module.

This matters for ML work in two ways. Large NumPy or tensor buffers are freed the moment the last reference disappears, so holding a stray reference (a list that keeps every batch's loss *tensor*, say) silently keeps memory alive. And in an autograd engine, the graph is a web of objects that reference each other, so it is the references you keep, not the ones you create, that decide how much memory a training step holds.

### The GIL

The **Global Interpreter Lock** is a mutex that lets only one thread execute Python bytecode at a time. It exists because reference counts are not atomic: without it, two threads could corrupt a count. Consequences:

- **CPU-bound pure-Python code gains nothing from threads.** Two threads doing the same arithmetic take about as long as running the work twice in a row (you will measure this in the task).
- **I/O-bound code does gain**, because the GIL is released while waiting on the network or disk.
- **C extensions can release the GIL** while they work on raw memory. NumPy does this inside big operations, which is why threads can speed up NumPy-heavy code, and why `multiprocessing` or native code is the usual answer for Python-level parallelism.

(Recent CPython versions have an optional free-threaded build, but the GIL-based model above is still the default you will meet in practice.)

### Crossing into C: extensions and Cython

A **C extension** is compiled code that CPython can import. You can write one by hand against the C API (`PyObject*`, manual refcounting), with pybind11 (Sprint 4), or with **Cython**, which lets you write Python-looking code that is translated to C. The key is *typing*:

```cython
# cython: boundscheck=False, wraparound=False
def dot(double[::1] a, double[::1] b):
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s
```

`double[::1]` is a **typed memoryview** onto a contiguous buffer of C doubles (any NumPy array of `float64` qualifies). With `cdef`-typed `i` and `s`, the loop compiles to a plain C loop over unboxed numbers: no objects, no dispatch, no refcounting. `boundscheck=False` and `wraparound=False` remove per-access safety checks; you are trading safety for speed, so only do it once the code is verified.

Expect a pure-Python loop at roughly 30–40 ms per million elements, Cython near 1 ms, and `np.dot` around or below that (it calls an optimised BLAS routine that uses SIMD instructions). The gap between Python and Cython is the cost of boxing and dispatch; the smaller gap between Cython and NumPy is vectorisation and cache-aware code.

## Part 2. A tensor is flat memory plus bookkeeping

### The big idea

A 2-D "matrix" has no 2-D memory. Memory is a single line of bytes. A tensor is the combination of:

- a **storage**: one contiguous buffer of numbers;
- a **shape**: how many elements along each dimension;
- **strides**: how many *elements* to skip in the buffer to move one step along each dimension;
- an **offset**: where in the buffer this tensor's first element lives.

The element at index `(i, j, k)` is `storage[offset + i*stride0 + j*stride1 + k*stride2]`. That one formula is the whole indexing story.

### Row-major strides

For shape `(3, 4)` stored in **row-major (C) order**, the last dimension is contiguous, so strides are `(4, 1)`: stepping along a row moves 1 element, stepping down a column skips 4. In general `stride[last] = 1` and `stride[i] = stride[i+1] * shape[i+1]`. For shape `(2, 3, 4)` that gives `(12, 4, 1)`.

### Views: change the bookkeeping, not the data

Because the layout is described by shape/strides/offset, many operations can be done by changing only that description:

- **Transpose** of a `(3, 4)` tensor with strides `(4, 1)` is a `(4, 3)` tensor with strides `(1, 4)`. Same storage, no copying, O(1).
- **Select** (`t[1, :]` or `t[:, 2]`) adds `index * stride` to the offset and drops a dimension.
- **Reshape** of a contiguous tensor just recomputes row-major strides for the new shape.

All of these are **views**: they share storage, so writing through one changes the others. That is powerful (it makes slicing free) and dangerous (it creates aliasing bugs). The transposed tensor above is also **non-contiguous**: its strides no longer match row-major order for its shape, so walking it in index order jumps around memory. Many kernels require contiguous input, and some operations (like `reshape` after a transpose) are impossible as a view. That is why frameworks have a `contiguous()` method that makes a **fresh copy** in row-major order, which you then reshape freely.

Memory access order matters for speed, too. Reading a row-major matrix along its rows walks memory sequentially and is cache-friendly; reading down a column strides through memory and is not. This same idea returns in CUDA as *coalesced* memory access (Sprint 5).

### Ownership in C++: shared storage and RAII

If many views point at the same buffer, who frees it? In C++ the clean answer is **RAII with shared ownership**: the buffer lives in a `std::shared_ptr<float[]>`, every view copies the pointer (incrementing an atomic count), and the memory is freed exactly when the last view disappears. There is no manual `delete`, so the lifetime bug "the original tensor died but my view still points at its memory" cannot happen. You will test exactly that.

Design notes worth carrying forward:

- Separate `Storage` (owns memory) from `Tensor` (describes a window onto it). PyTorch does the same thing, which is why `tensor.untyped_storage()` exists.
- Make the default copy of a `Tensor` a *view* (cheap pointer copy) and make real copies explicit (`contiguous()`, `clone()`). Hidden deep copies are a classic performance trap.
- Bounds-check in `at()` while learning; frameworks skip it in hot loops and validate once at the boundary.

## What to take into the task

1. Python is slow per element because of boxing, dynamic dispatch and refcounting, not because bytecode is "interpreted" in some mystical way. Fix it by moving the loop to typed C code.
2. The GIL makes threads useless for CPU-bound Python, but not for I/O or for C code that releases it.
3. A tensor is `(storage, shape, strides, offset)`. Views change the last three; copies change the first.
4. Contiguity is a property of strides, and many operations need it.

## Further reading

- Edward Yang, *PyTorch internals*: http://blog.ezyang.com/2019/05/pytorch-internals/
- NumPy docs, *Internal memory layout of an ndarray*: https://numpy.org/doc/stable/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray
- David Beazley, *Understanding the Python GIL*: https://www.dabeaz.com/GIL/
- Python docs, `dis`: https://docs.python.org/3/library/dis.html
- Cython docs, typed memoryviews: https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html
