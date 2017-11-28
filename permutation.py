def permutation(n):
    """Return all of the permutations of n elements.

    Args:
    n: number of elements

    Returns:
    all of the permutations of n elements
    """
    numbers = [i for i in range(n)]
    return list(permutation_generator(numbers))


def permutation_generator(array, perm=[]):
    """Generate a permutation of an array.

    Args:
    array: array whose permutation to generate
    perm: permutation of array (You mustn't pass this argument)

    Yields:
    temp_perm: permutation of array
    """
    for i in range(len(array)):
        temp_array = array[:]
        temp_perm = perm[:]
        temp_perm.append(temp_array.pop(i))
        if temp_array != []:
            yield from permutation_generator(temp_array, temp_perm)
        else:
            yield temp_perm
