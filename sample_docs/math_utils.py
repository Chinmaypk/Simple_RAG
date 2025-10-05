def add_numbers(a, b):
    """
    Returns the sum of two numbers.
    """
    return a + b


def factorial(n):
    """
    Computes the factorial of n using recursion.
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def is_prime(n):
    """
    Checks if a number is prime.
    """
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
