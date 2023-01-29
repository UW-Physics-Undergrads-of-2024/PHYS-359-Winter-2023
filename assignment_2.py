from concurrent.futures import ThreadPoolExecutor
from numba import njit
from numpy import array, empty, exp


@njit(nogil=True)
def partition_function(beta_epsilon: float) -> float:
    """
    Numerically calculate the value of the partition function
    :param beta_epsilon: The product of thermodynamic beta (i.e. coldness) and epsilon, the energy state
    :return: the value of the partition function
    """

    # Declare an array without initializing it. This is done so that Numba's JIT compiler knows the size of the array
    # at compile time
    summand_values = empty(101)

    # Initialize values of the array
    for row, value in enumerate(summand_values):
        summand_values[row] = (2 * row + 1) * exp(-beta_epsilon * row * (row + 1))

    # Calculate the value of the partition function
    Z = sum(summand_values)
    return Z


if __name__ == '__main__':
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(partition_function, array([0.01, 0.2, 0.5, 1])))
    print(f"For beta * epsilon = 0.01, the partition function's value is: {results[0]}")
    print(f"For beta * epsilon = 0.2, the partition function's value is: {results[1]}")
    print(f"For beta * epsilon = 0.5, the partition function's value is: {results[2]}")
    print(f"For beta * epsilon = 1, the partition function's value is: {results[3]}")
