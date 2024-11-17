"""Collection of the core mathematical operators used throughout the code base."""

import math


# ## Task 0.1
from typing import Callable, Iterable, List, TypeVar

# Implementation of a prelude of elementary functions.
# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The product of `x` and `y`.

    """
    return float(x * y)


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The input number `x` unchanged.

    """
    return float(x)


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The sum of `x` and `y`.

    """
    return float(x + y)


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The negated value of `x`.

    """
    return float(-1 * x)


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: `True` if `x` is less than `y`, otherwise `False`.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: `True` if `x` is equal to `y`, otherwise `False`.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The larger of `x` and `y`.

    """
    return float(x) if x > y else float(y)


def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Checks if two numbers are close in value within a tolerance.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.
        tol (float, optional): The tolerance level. Defaults to 1e-9.

    Returns:
    -------
        bool: `True` if `x` and `y` are close within the tolerance, otherwise `False`.

    """
    return abs(x - y) < tol


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The sigmoid of `x`, defined as 1 / (1 + exp(-x)).

    """
    if x >= 0:
        res = 1 / (1 + math.exp(-x))
    else:
        res = math.exp(x) / (1 + math.exp(x))
    return float(res)


def relu(x: float) -> float:
    """Applies the ReLU (Rectified Linear Unit) activation function.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The ReLU of `x`, which is `x` if `x > 0`, otherwise `0`.

    """
    if x > 0:
        res = x
    else:
        res = 0
    return float(res)


def log(x: float) -> float:
    """Calculates the natural logarithm of a number.

    Args:
    ----
        x (float): The input number. Must be positive.

    Returns:
    -------
        float: The natural logarithm of `x`.

    Raises:
    ------
        ValueError: If `x` is less than or equal to 0.

    """
    if x <= 0:
        raise ValueError("Input must be positive.")
    return float(math.log(x))


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The exponential of `x`, defined as e^x.

    """
    return float(math.exp(x))


def inv(x: float) -> float:
    """Calculates the reciprocal of a number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The reciprocal of `x`, defined as 1 / x.

    Raises:
    ------
        ZeroDivisionError: If `x` is zero.

    """
    if x == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return float(1 / x)


def log_back(x: float, a: float) -> float:
    """Computes the derivative of the logarithm function times a second argument.

    Args:
    ----
        x (float): The input to the logarithm function.
        a (float): A constant to multiply to the derivate of the log function

    Returns:
    -------
        float: The derivative of the logarithm function times `a`.

    Raises:
    ------
        ValueError: If `x` is less than or equal to 0.

    """
    if x <= 0:
        raise ValueError("Input must be positive.")
    return float(a / x)


def inv_back(x: float, a: float) -> float:
    """Computes the derivative of the reciprocal function times a second argument.

    Args:
    ----
        x (float): The input to the reciprocal function.
        a (float): The derivative of the output with respect to the reciprocal's output.

    Returns:
    -------
        float: The derivative of the reciprocal function times `a`.

    """
    return float(-a / (x**2))


def relu_back(x: float, a: float) -> float:
    """Computes the derivative of the ReLU function times a second argument.

    Args:
    ----
        x (float): The input to the ReLU function.
        a (float): The derivative of the output with respect to ReLU's output.

    Returns:
    -------
        float: The derivative of ReLU function times `a`.

    """
    if x > 0:
        der = 1
    else:
        der = 0

    return float(a * der)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")
X = TypeVar("X")
N = TypeVar("N", float, int)


def map(func: Callable[[T], U], iterable: Iterable[T]) -> List[U]:
    """Applies a given function to each element of an iterable.

    Args:
    ----
        func (Callable[[T], U]): A function that takes an element of type `T` and returns a value of type `U`.
        iterable (Iterable[T]): An iterable of elements of type `T`.

    Returns:
    -------
        List[U]: A list of elements of type `U` obtained by applying `func` to each element of `iterable`.

    """
    return [func(x) for x in iterable]


def zipWith(
    func: Callable[[T, U], V], iterable1: Iterable[T], iterable2: Iterable[U]
) -> List[V]:
    """Combines elements from two iterables using a given function.

    Args:
    ----
        func (Callable[[T, U], V]): A function that takes two elements (one from each iterable) and returns a value of type `V`.
        iterable1 (Iterable[T]): The first iterable of elements of type `T`.
        iterable2 (Iterable[U]): The second iterable of elements of type `U`.

    Returns:
    -------
        List[V]: A list of elements of type `V` obtained by applying `func` to each pair of elements from `iterable1` and `iterable2`.

    """
    return [func(x, y) for x, y in zip(iterable1, iterable2)]


def reduce(func: Callable[[T, T], T], iterable: Iterable[T]) -> T:
    """Reduces an iterable to a single value using a given function.

    Args:
    ----
        func (Callable[[T, T], T]): A function that takes two elements of type `T` and returns a single element of type `T`.
        iterable (Iterable[T]): An iterable of elements of type `T`.
        initializer (T, optional): A starting value to initialize the reduction. Defaults to None.

    Returns:
    -------
        T: A single value of type `T` obtained by reducing `iterable` using `func`.

    """
    it = iter(iterable)
    value = next(it)
    for element in it:
        value = func(value, element)
    return value


def negList(lst: List[N]) -> List[N]:
    """Negates all elements in a list using the `map` function.

    Args:
    ----
        lst (List[float]): A list of floats.

    Returns:
    -------
        List[float]: A list with all elements negated.

    """
    return map(lambda x: -x, lst)


def addLists(lst1: List[N], lst2: List[N]) -> List[N]:
    """Adds corresponding elements from two lists using the `zipWith` function.

    Args:
    ----
        lst1 (List[float]): The first list of floats.
        lst2 (List[float]): The second list of floats.

    Returns:
    -------
        List[float]: A list of floats where each element is the sum of the corresponding elements in `lst1` and `lst2`.

    """
    return zipWith(lambda x, y: x + y, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Sums all elements in a list using the `reduce` function.

    Args:
    ----
        lst (List[float]): A list of floats.

    Returns:
    -------
        float: The sum of all elements in `lst`.

    """
    if len(lst) == 0:
        res = 0
    else:
        res = reduce(lambda x, y: x + y, lst)
    return float(res)


# def prod(lst: List[N]) -> N:
def prod(lst: Iterable[N]) -> N:
    """Calculates the product of all elements in a list using the `reduce` function.

    Args:
    ----
        lst (List[float]): A list of floats.

    Returns:
    -------
        float: The product of all elements in `lst`.

    """
    return reduce(lambda x, y: x * y, lst)
