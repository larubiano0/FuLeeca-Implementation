# FuLeeca (Python 3) — Research Demo & Reproducible Scripts

A **research-grade** Python implementation of the core flow behind [FuLeeca](https://eprint.iacr.org/2023/377), a code-based digital signature scheme over the Lee metric. This repo accompanies the author’s thesis and demonstrates:

- Key generation using circulant (shift) matrices built from vectors sampled uniformly from a Lee sphere

- Signature generation via a “simple sign” step plus a concentration routine that enforces weight and security thresholds

- Signature verification using a public generator matrix and a reproducible per-message challenge

> ⚠️ Security disclaimer
This code is for educational/research use. It is not a production implementation and has not undergone cryptographic review or side-channel hardening. Parameters and thresholds mirror the thesis, but you should not deploy this as a security-critical system.

> **Reference C implementation:**  
> The original C implementation of FuLeeca is available at  
> [https://gitlab.lrz.de/tueisec/fuleeca-signature/-/tree/main/Reference_Implementation/FuLeeca1](https://gitlab.lrz.de/tueisec/fuleeca-signature/-/tree/main/Reference_Implementation/FuLeeca1)

## Contents:

```bash
├── key_generation.py          # Build (G_sec, G_pub) for Parameter Choice I
├── signature_generation.py    # simple_sign + concentrate loop
├── signature_verification.py  # rank-based code-membership + weight + LMP checks
├── utils.py                   # Lee/Hamming weights, LMP, Lee-sphere sampler, shift matrix
├── main.py                    # End-to-end demo (keys → signature → verification)
├── requirements.txt
├── keys_p65521_n1318_wk31102.pkl    # (optional) cached keys
└── sig_p65521_n1318_wk31102.pkl     # (optional) cached signature
```

## Quick start

### 1) Environment

```bash
    python3 -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
```
> Tested with `galois==0.4.6` and `numpy==2.2.6`.

### 2) Run the demo

```bash
    python main.py
```

First run will:
- Generate keys (can take a while), then cache them to `keys_p65521_n1318_wk31102.pkl`
- Sign the message `"Hello, FuLeeca!"` and cache to `sig_p65521_n1318_wk31102.pkl`
- Verify the signature with the public key

Subsequent runs will load the cached artifacts.

### Parameter choice I

| Symbol | Meaning                                   | Value (default) |
|-------:|-------------------------------------------|-----------------:|
| `p`    | Prime field size                          | `65521`          |
| `n`    | Code length (must be even)                | `1318`           |
| `k`    | `n/2`                                     | `659`            |
| `w_key`| Lee weight of secret-key vector(s)        | `31102`          |
| `w_sig_over_n` | Target average Lee weight per coordinate | `982.8`    |
| `w_sig`| Signature Lee-weight bound (`w_sig_over_n * n`) | computed  |
| `s`    | Scaling factor in `simple_sign`           | `3/64`           |
| `n_con`| Concentration iterations                  | `100`            |
| `sec_level` | Security level (bits)                | `160`            |

### Performance and reproducibility

- **Key generation**: uses a DP-based uniform sampler over Lee spheres (`utils._build_dp`).
For large (`p, k, t`) this can be slow and memory-intensive (several days and GB of RAM). The repo caches keys/signatures as .pkl.

- **Randomness**: 
  - `signature_generation` uses `secrets.token_bytes(32)` and `SHAKE-256` for per-message challenges.
  - The Lee-sphere sampler uses Python’s random internally. You may set `random.seed(...)` for reproducibility, but doing so is not appropriate for real cryptography.

### Troubleshooting

1) Unpickling `galois` field arrays

When loading cached keys, you may see an error like:

```bash
AttributeError: module 'galois._fields._factory' has no attribute 'FieldArray_65521_17'
```

This stems from how galois names dynamic field classes at runtime. The demo handles this by registering the class name prior to unpickling:

```python
factory = importlib.import_module("galois._fields._factory")
GF = galois.GF(p)
setattr(factory, "FieldArray_65521_17", GF)
```

If your error shows a **different** suffix, update that line with the exact name from the error message (printed by `galois` on your machine/version).

2) Long key-gen times

Key generation may take a long time for these parameters. That’s expected for a Python prototype that prioritizes clarity and correctness over speed. Consider keeping the cached .pkl files.

3) Verification fails

- Ensure you’re using the same `m`, `salt`, and `G_pub` produced during signing.
- Check that `w_sig_over_n` and other public parameters in `main.py` match signing time.

### API overview

```python
key_generation.generate_key(p: int, n: int, w_key: int) -> (G_sec, G_pub)
```

- Samples `a`,`b` uniformly from the Lee sphere of radius `w_key`, builds `A=Shift(a)`, `B=Shift(b)`, returns:
    - `G_sec ∈ F_p^{k×n}, G_pub ∈ F_p^{k×k}`.

```python
signature_generation.signature_generation(G_sec, n, m, w_sig, w_key, s, n_con, sec_level, p) -> (v, salt)
```
- Full signing routine: prehash+SHAKE challenge → `simple_sign` → `concentrate`.

```python
signature_verification.signature_verification(v, n, G_pub, m, w_sig, sec_level, salt, p) -> bool
```

- Rebuilds `c` from `(m, salt)`, checks code membership, Lee-weight, and `LMP`.

```python
utils.py
```
- Helper functions: Lee/Hamming weights, `mt(·,·)` sign-match count, `LMP(·,·)`, uniform Lee-sphere sampler, and `shift_matrix`.

### Extending / experimenting

- **Different parameter sets**: adjust `(p, n, w_key, w_sig_over_n, s, n_con, sec_level)` in `main.py`.
- **Alternative samplers**: swap `sample_x_uniformly_from_Lee_sphere` for a faster (approximate) method if you’re exploring speed/accuracy trade-offs.
- **Heuristics in concentrate**: the acceptance rule and the target `LMP` can be tuned for experiments.


### Acknowledgements

- Built with the excellent [`galois`](https://mhostetter.github.io/galois/latest) library for finite field arithmetic (thanks to Matt Hostetter).

- The scheme structure, thresholds, and parameterization follow FuLeeca as described in [the paper](https://eprint.iacr.org/2023/377) and [the author’s thesis](https://google.com).


### License

No license file is currently included. If you intend to share/modify, please add an explicit license (e.g., MIT/BSD/Apache-2.0) at the root of the repo.

### Citation

If you use this code in academic work, please cite the accompanying thesis (author: Luis Alejandro Rubiano Guerrero). If you publish a bib entry, add a link/DOI to the PDF once available.