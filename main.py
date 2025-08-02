#  ─────────────────────────────────────────────────────────────────────
#  Full demo for FuLeeca – Parameter Choice I
#  ─────────────────────────────────────────────────────────────────────
from pathlib import Path
import pickle
from key_generation import generate_key
# from signature_generation   import signature_generation
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

if key_file.is_file():
    # ----------  Load existing keys  ----------
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


'''
# --------------------------------------------------------------------
# 3. Sign a message
# --------------------------------------------------------------------
message           = b"Hello, FuLeeca!"
salt, sig_bytes   = signature_generation(
                        message, G_sec,
                        p=p, wsig=w_sig, wkey=w_key, _lambda=lam)

print("signature bytes length :", len(sig_bytes))
print("salt (hex)             :", salt.hex()[:32] + "...")

# --------------------------------------------------------------------
# 4. Verify the signature
# --------------------------------------------------------------------
valid = signature_verification(
            message, salt, sig_bytes, T,
            p=p, wsig=w_sig, _lambda=lam)

print("verification result    :", "VALID" if valid else "INVALID")
'''