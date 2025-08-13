import galois
import numpy as np
import utils 
import hashlib
import secrets

def signature_generation(G_sec, n, m, w_sig, w_key, s, n_con, sec_level, p):
    """
    Generate a signature for a message using the secret key.
    
    Parameters
    ----------
    G_sec : galois.FieldArray
        The secret key matrix of shape (n/2, n).
    n : int
        The code length.
    m : int
        The number of iterations for concentration.
    w_sig : float
        The target Lee weight per codeword.
    w_key : int
        The Lee weight of the secret key vector.
    s : float
        Scaling factor for the signature generation.
    n_con : int
        Iteration budget for concentration.
    sec_level : int
        Security level in bits.
    p : int
        The size of the finite field, which is a prime number.

    Returns
    -------
    v : galois.FieldArray
        The generated signature vector.
    salt : bytes
        Random salt used in the signature generation.
    """
    k = n // 2
    Hash = hashlib.sha3_512()
    Hash.update(m) 
    m_prime = Hash.digest()[:(2 * sec_level)//8] # Hash the message and truncate to security level

    while True:
        v, c, salt = simple_sign(m_prime, n, k, G_sec, s, p)
        v = concentrate(v, c, k, G_sec, n_con, w_sig, w_key, sec_level, p)
        print(w_sig - 2*w_key, utils._Lee_weight(v, p), w_sig, utils.LMP(v, c, p), sec_level + 64)##
        if (w_sig - 2*w_key <= utils._Lee_weight(v, p) <= w_sig) and utils.LMP(v, c, p) >= sec_level + 64:
            return v, salt


def simple_sign(m_prime, n, k, G_sec, s, p):
    """
    Generate a simple signature vector based on the message and secret key.
    
    Parameters
    ----------
    m_prime : bytes
        The message to be signed, prehashed.
    n : int
        The code length.
    k : int
        The code length over two.
    G_sec : galois.FieldArray
        The secret key matrix.
    s : float
        Scaling factor for the signature generation.
    p : int
        The size of the finite field, which is a prime number.

    Returns
    -------
    v : galois.FieldArray
        The generated signature vector.
    c : galois.FieldArray
        The codeword used in the signature generation.
    salt : bytes
        Random salt used in the signature generation.
    """
    salt = secrets.token_bytes(32) # Sample from {0, 1}^256
    CSPRNG = hashlib.shake_256()
    CSPRNG.update(m_prime + salt)
    GF = galois.GF(p)
    c = np.unpackbits(np.frombuffer(CSPRNG.digest(n//8 + 1), dtype=np.uint8))[:n] # Numpy array of shape (n,) with entries in {0,1}, the +1 and the slicing is to ensure we have enough bits
    c = [((-1)**int(bit))%p for bit in c] # Convert to {+1, -1}
    c = GF(c) # Convert to galois.FieldArray
    x = GF([0]*k)
    for i in range(k):
        g_i = G_sec[i,:]
        x[i] = (np.trunc((utils.mt(g_i, c, p) - utils.Hamming_weight(g_i*c, p)/2)*s).astype(np.int64)) % p
    v = x @ G_sec
    return v, c, salt


def concentrate(v, c, k, G_sec, n_con, w_sig, w_key, sec_level, p):
    """
    Concentrate the signature vector to meet the required Lee weight and security level.
    
    Parameters
    ----------
    v : galois.FieldArray
        The initial signature vector.
    c : galois.FieldArray
        The codeword used in the signature generation.
    k : int
        The code length over two.
    G_sec : galois.FieldArray
        The secret key matrix.
    n_con : int
        Iteration budget for concentration.
    w_sig : float
        The target Lee weight per codeword.
    w_key : int
        The Lee weight of the secret key vector.
    sec_level : int
        Security level in bits.
    p : int
        The size of the finite field, which is a prime number.

    Returns
    -------
    v_concentrated : galois.FieldArray
        The concentrated signature vector.
    """
    A = list(range(1, k+1)) + list(range(-k, 0)) # Indexes {1, ..., k, -k, ..., -1}
    lf = 1

    GF = galois.GF(p)
    for _ in range(n_con):
        v_prime = GF([0]*k)
        i = 1
        centinel = False
        while i != -k: # Iterate over {1, -1, ..., k, -k}
            g_i = G_sec[np.abs(i)-1, :] # Get the |i|-th row of G_sec
            v_prime_prime = v + g_i if i > 0 else v - g_i
            if (np.abs(utils.LMP(v_prime_prime, c, p) - (sec_level + 65)) <= np.abs(utils.LMP(v_prime, c, p) - (sec_level + 65))) and ((i in A) or lf == 0):
                v_prime = v_prime_prime
                i_prime = i
                centinel = True
            if i > 0:
                i = -i
            else:
                i = -i + 1
        if centinel:
            v = v_prime
            A.remove(-i_prime)
        if utils._Lee_weight(v, p) > w_sig - w_key:
            lf = 0
        else:
            lf = 1
    return v
            