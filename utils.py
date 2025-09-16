import galois
import numpy as np
import random
from math import lgamma, log
from functools import lru_cache  
import sys

sys.setrecursionlimit(10_000) 


def _Lee_weight(x, p):
    """
    Compute the Lee weight of a vector x in F_p^k.
    The Lee weight is defined as the sum of the absolute values of the elements
    in the vector, where each element is considered modulo p.

    Parameters
    ----------
    x : galois.FieldArray 
        A vector in the finite field F_p^k.
    p : int
        The size of the finite field, which is a prime number.

    Returns
    -------
    int
        The Lee weight of the vector x, which is the sum of the minimum
        of the absolute value of each element and p minus the absolute value
        of each element, effectively counting the "distance" in the Lee metric.
    """
    sumation = 0
    for xi in x:
        xi = int(xi)
        if xi < p-xi:
            sumation += xi
        else:
            sumation += p - xi
    return sumation

def Hamming_weight(x, p):
    """
    Compute the Hamming weight of a vector x in F_p^k.
    The Hamming weight is defined as the number of non-zero elements in the vector.

    Parameters
    ----------
    x : galois.FieldArray 
        A vector in the finite field F_p^k.
    p : int
        The size of the finite field, which is a prime number.

    Returns
    -------
    int
        The Hamming weight of the vector x, which is the count of non-zero elements.
    """
    return np.sum(1 for xi in x if xi != 0)


def mt(x, y, p):
    """
    Compute the sign matches between two vectors x and y.

    Parameters
    ----------
    x : galois.FieldArray 
        The first vector.
    y : galois.FieldArray 
        The second vector.
    p : int
        The size of the finite field, which is an odd prime number.
    Returns
    -------
    int
        The number of positions where the signs of the elements in x and y match.
    """
    sumation = 0
    for xi, yi in zip(x, y):
        if xi == 0 or yi == 0:
            continue
        sign_xi = 1 if xi <= (p-1)//2 else -1
        sign_yi = 1 if yi <= (p-1)//2 else -1
        if sign_xi == sign_yi:
            sumation += 1
    return sumation


def LMP(v, c, p):
    """
    Compute the Logaritmic Matching Probability (LMP) between two vectors v and c.

    Parameters
    ----------
    v : galois.FieldArray 
        The first vector.
    c : galois.FieldArray 
        The second vector.
    p : int
        The size of the finite field, which is an odd prime number.
    Returns
    -------
    float
        The Logarithmic Matching Probability (LMP) between the two vectors.
    """
    xv = np.asarray(v, dtype=np.int64) % p
    yv = np.asarray(c, dtype=np.int64) % p

    nz = (xv != 0) & (yv != 0)
    k = int(nz.sum())
    if k == 0:
        return 0.0

    xsgn = np.where(xv[nz] <= (p - 1)//2, 1, -1)
    ysgn = np.where(yv[nz] <= (p - 1)//2, 1, -1)
    mu = int((xsgn == ysgn).sum())

    # LMP = k - log2 C(k, mu)
    log2C = (lgamma(k+1) - lgamma(mu+1) - lgamma(k-mu+1)) / log(2.0)
    return float(k - log2C)


def _weighted_choice_int(population, weights):
    """
    Performs a weighted random choice on a population.

    This function is designed to work with arbitrarily large integer weights,
    avoiding the OverflowError that occurs when `random.choices` tries to
    convert huge weights to floats.

    Parameters
    ----------
    population : list
        The list of items to choose from.
    weights : list
        A list of integer weights, corresponding to the population.

    Returns
    -------
    Any
        A single element chosen from the population.
    """
    total_weight = sum(weights)
    if total_weight == 0:
        # Or handle this case as you see fit, maybe raise an error
        # if all weights are zero.
        return random.choice(population)

    r = random.randrange(total_weight)
    upto = 0
    for i, w in enumerate(weights):
        if upto + w > r:
            return population[i]
        upto += w
    assert False, "Should not get here"


@lru_cache(maxsize=None)
def _build_dp(k, t, M):
    """
    Construct a dynamic-programming table that answers the question:

        How many length-l suffixes over the alphabet {0, 1, ..., M}
        have total Lee weight w, when every non-zero symbol may be
        chosen with either sign?

    The table entry dp[l][w] stores that count for every
    0 <= l <= k and 0 <= w <= t.  Row l = 0 is initialised so that
    dp[0][0] = 1 and dp[0][w] = 0 for w > 0.

    Each subsequent row is filled by a convolution with the kernel

        [1, 2, 2, ..., 2]      (length M + 1)

    where 1 corresponds to choosing the symbol 0 in the new
    coordinate and 2 accounts for the two possible signs of each
    non-zero symbol. 

    Parameters
    ----------
    k : int
        Full vector length.
    t : int
        Target Lee weight.
    M : int
        Maximum absolute symbol value (p-1/2 for a prime field F_p).

    Returns
    -------
    numpy.ndarray
        A (k + 1) by (t + 1) array of Python integers whose element
        [l, w] is the count described above.
    """
    dp = np.zeros((k + 1, t + 1), dtype=object)
    dp[:, 0] = 1                      # weight 0 achievable with any length

    kernel = np.ones(M + 1, dtype=object)
    kernel[1:] *= 2                  # sign for non-zero addends

    for ell in range(1, k + 1):
        if ell%10 == 0:
            print(f"Building DP table: {ell}/{k} rows filled", end='\r')
        prev = dp[ell - 1]
        curr = dp[ell]
        # convolution via polynomial multiplication
        curr[:] = prev
        for a in range(1, M + 1):
            curr[a:] += 2 * prev[:-a]
    return dp

'''
Removed because ennumerating partitions is not efficient for large t. 
Replaced with a dynamic programming approach in _build_dp.
@lru_cache(maxsize=None) 
def _restricted_partitions_cached(t, r):
    parts = []
    for d in partitions(t, k=r): # k bounds the largest part
        flat = [part for part, mult in d.items() for _ in range(mult)]
        parts.append(tuple(sorted(flat, reverse=True)))
    return tuple(parts)                  # hashable return value


def _restricted_partitions(t, r):
    """
    Generate every partition of t whose parts are <= r, using SymPy.
    Each partition is returned as a descending list of ints.

    Parameters
    ----------
    t : int
        The integer to be partitioned.
    r : int
        The maximum value for the parts of the partition.

    Yields
    ------
    list
        A list representing a partition of t, where each part is less than or equal to r.
        The parts are sorted in descending order.
    """
    for lam in _restricted_partitions_cached(t, r):
        yield list(lam)

@lru_cache(maxsize=None)                
def _partition_weight_cached(lam_tuple, k):
    l = len(lam_tuple)
    comb = math.comb(k, l)
    perm = math.factorial(l)
    mult = math.prod(math.factorial(m) for m in Counter(lam_tuple).values())
    return comb * perm // mult * (1 << l) # (1 << l) is 2^l


def _partition_weight(lam, k):
    """
    Compute |V_{t,lam}^{(k)}|  =  (k choose l) * l! / prod(mult_i!) * 2^l,
    where l = len(lam) and mult_i are the multiplicities of equal parts.

    Parameters
    ----------
    lam : list
        A list representing a partition of an integer t.
    k : int
        The length of the vector space F_p^k.   
        
    Returns
    -------
    int
        The number of vectors in the Lee sphere S_L(p, t, k) corresponding to
        the partition lam, where t is the sum of the parts in lam.
        This is the size of the set of vectors with Lee weight equal to the sum of the
        parts in lam, and each part is considered modulo p.
    """
    return _partition_weight_cached(tuple(lam), k)
'''

def sample_x_uniformly_from_Lee_sphere(p, k, t):
    '''
    Sample x in F_p^k uniformly from the Lee sphere
        S_L(p, t, k) = { v in F_p^k : w_L(v) = t }.
        without enumerating integer partitions.

    Parameters
    ----------
    p : int    – prime field size (odd in typical use).
    t : int    – target Lee weight.
    k : int    – vector length.

    Returns
    -------
    x : galois.FieldArray 
        A vector sampled uniformly from the Lee sphere.
    '''
    M  = p // 2
    GF = galois.GF(p)
    dp = _build_dp(k, t, M)

    x, w_left = [0]*k, t
    for i in range(k):
        tail = k - i - 1
        probs = []
        choices = range(min(M, w_left) + 1)
        for a in choices:
            ways = dp[tail, w_left - a]
            probs.append(ways * (2 if a else 1))
        a = _weighted_choice_int(list(choices), probs)
        if a:
            x[i] = (random.choice((-1,1))*a) % p
        w_left -= a
    assert _Lee_weight(x, p) == t, f"Sampled vector {x} has Lee weight {_Lee_weight(x, p)} != {t}"
    return GF(x)


def shift_matrix(x):
    """
    Build the k x k circulant-like matrix Shift(x), where
    each row is a left cyclic shift of the 1-D field array x.

    Parameters
    ----------
    x : galois.FieldArray, shape (k,)
        Input vector over GF(p) or GF(p^m).

    Returns
    -------
    galois.FieldArray, shape (k, k)
        The matrix shown in the paper.
    """
    k = x.size
    GF = type(x)                                  # the field of x
    M = GF.Zeros((k, k))                          # allocate in that field
    for i in range(k):
        M[i, :] = np.roll(x, i)                  # cyclic left shift by i
    return M


# Example usage of sample_x_uniformly_from_Lee_sphere
#random.seed(42)       # for reproducibility
#print(sample_x_uniformly_from_Lee_sphere(p=7, k=4, t=5))
#print(sample_x_uniformly_from_Lee_sphere(p=101, k=50, t=500))

# Example usage of shift_matrix
# GF = galois.GF(7)
# a  = GF([1, 2, 3, 4])
# S = shift_matrix(a)
# print(S)