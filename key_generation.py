import galois
import numpy as np
import utils

def generate_key(p, n, w_key):
    """
    Generate a secret key and a public key for the FuLeeca signature scheme.
    
    Parameters
    ----------
    p : int
        The size of the finite field, which is a prime number.
    n : int
        The code length, which should be even.
    w_key : int
        The fixed Lee weight of the secret-key vector.
    Returns
    -------
    G_sec : np.ndarray
        The secret key matrix of shape (n/2, n).
    G_pub : np.ndarray
        The public key matrix of shape (n/2, n/2).

    """
    GF = galois.GF(p)
    while True:
        a = utils.sample_x_uniformly_from_Lee_sphere(p, n//2, w_key)
        A = utils.shift_matrix(a)
        if np.linalg.det(A) != 0: # ensure A is invertible mod p
            break
    b = utils.sample_x_uniformly_from_Lee_sphere(p, n//2, w_key)
    B = utils.shift_matrix(b)
    G_sec = GF(np.hstack((A, B)))  # secret key
    G_pub = GF(np.hstack((GF.Identity(n//2), np.linalg.inv(A) @ B)))  # public key
    return G_sec, G_pub