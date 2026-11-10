## Started Day 1 of 100 days of ML by WiMLDS and WID Nairobi Chapters

##Installation
`pip install Cython`

# Day 1: Python internals & C extensions (≈3 h)
## Goal: Basics of Cython
1. Create a simple python file such as `helloworld.py` or `fibonacci.py`
2. Create a setup file ie `setup.py`
3. run `python setup.py build_ext --inplace`
4. start python interpreter on the code editor or on terminal by typing python. 

You should see >>> when it's started
5. import the python file as a python module ie
>>>`import helloworld`


## Goal: understand what CPython does when it runs your code, and write a C-speed function you can call from Python.
Learn (≈1.5 h)
• David Beazley, Understanding the Python GIL (watch at 1.25x) and his GIL materials
• dis module docs: run dis.dis on a small function and read the bytecode line by line
• Cython tutorial and typed memoryviews
## Goal: Extending and Embedding Python and Anthony Shaw's CPython Internals
Build (≈1.5 h): vector dot product, three ways
1. Pure Python loop.
2. Cython with typed memoryviews:
### dot.pyx
### cython: boundscheck=False, wraparound=False
`def dot(double[::1] a, double[::1] b):
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s`
Build it with cythonize -i dot.pyx, or a small setup.py using cythonize("dot.pyx"). 3. np.dot as the baseline.
Then two short experiments: (a) sys.getrefcount and gc.collect() on a few objects to see reference counting; (b) time a CPU-bound function on 1 thread vs 2 threads and observe the GIL stop the speed-up.

# Done when: 
1. you have a table of timings (Python vs Cython vs NumPy, n = 10⁶) 
2. can explain in three sentences why Cython is faster and what the GIL does to threads.


References
1. https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html#the-basics-of-cython
2. https://www.youtube.com/watch?v=Obt-vMVdM8s: Understanding the Python GIL
3. https://docs.python.org/3/library/dis.html: dis module docs
