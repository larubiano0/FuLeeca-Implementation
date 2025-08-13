#  ─────────────────────────────────────────────────────────────────────
#  Full demo for FuLeeca – Parameter Choice I
#  ─────────────────────────────────────────────────────────────────────
from pathlib import Path
import pickle
import galois
import importlib
from key_generation import generate_key
from signature_generation   import signature_generation

# from signature_verification import signature_verification

# ── 1.  Scheme parameters (Parameter Choice I) ───────────────────────
p          = 65_521        # prime field size
n          = 1_318         # code length
w_key      = 31_102        # Lee weight of secret‑key vector
w_sig_over_n = 982.8       # w_sig / n
s          = 3 / 64        # scaling factor
n_con      = 100           # iteration budget for concentration
sec_level  = 160           # security level

# ── 2.  Key generation / loading ────────────────────────────────

key_file = Path(f"keys_p{p}_n{n}_wk{w_key}.pkl")
sig_file = Path(f"sig_p{p}_n{n}_wk{w_key}.pkl")

if key_file.is_file():
    # ----------  Load existing keys  ----------
    factory = importlib.import_module("galois._fields._factory")
    GF = galois.GF(p)
    setattr(factory, "FieldArray_65521_17", GF) # THIS MIGHT CHANGE IF YOU GENERATE THE KEY AGAIN, IN GENERAL, COPY THE NAME FROM THE ERROR MESSAGE
    with key_file.open("rb") as fh:
        G_sec, T = pickle.load(fh)
    print(f"Loaded keys from {key_file}")
else:
    # ----------  Generate and store keys  ----------
    print("Generating keys - this can take a while…")
    G_sec, T = generate_key(p=p, n=n, w_key=w_key)

    with key_file.open("wb") as fh:
        pickle.dump((G_sec, T), fh)
    print(f"Keys saved to {key_file}")

print("secret key G_sec shape:", G_sec.shape)   # (n/2, n)
print("public key  T     shape:", T.shape)      # (n/2, n/2)


# ── 3.  Signature generation ────────────────────────────────

if sig_file.is_file():
    # ----------  Load existing signature  ----------
    with sig_file.open("rb") as fh:
        v, salt = pickle.load(fh)
    print(f"Loaded signature from {sig_file}")
else:
    # ----------  Generate and store signature  ----------
    print("Generating signature - this can take a while…")
    w_sig = w_sig_over_n * n
    m = "Hello, FuLeeca!".encode("utf-8")  # Bytes
    v, salt = signature_generation(
        G_sec=G_sec, n=n, m=m, w_sig=w_sig, w_key=w_key, s=s, n_con=n_con, sec_level=sec_level, p=p
    )
    with sig_file.open("wb") as fh:
        pickle.dump((v, salt), fh)
    print(f"Signature saved to {sig_file}")

print("signature vector v shape:", v.shape)  # (n/2,)
print("signature vector v:", v)

'''

# --------------------------------------------------------------------
# 4. Verify the signature
# --------------------------------------------------------------------
valid = signature_verification(
            message, salt, sig_bytes, T,
            p=p, wsig=w_sig, _lambda=lam)

print("verification result    :", "VALID" if valid else "INVALID")
'''