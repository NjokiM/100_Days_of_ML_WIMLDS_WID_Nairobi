def fib(n):
    "Print the Fibonacci series upto n"
    """In mathematics, the Fibonacci sequence is a sequence in which each element 
    is the sum of the two elements that precede it. Numbers that are part of the 
    Fibonacci sequence are known as Fibonacci numbers, commonly denoted Fn . 
    The initial elements of the sequence are F1 = 1 and F2 = 1, 
    though many authors also include a zeroth element F0 = 0.[1] Starting from F0, the sequence begins
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144"""

    a,b = 0,1
    while b < n:
        print(b, end=' ')
        a, b = b, a+b
    print()