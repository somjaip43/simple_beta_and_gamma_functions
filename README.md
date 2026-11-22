# simple_beta_and_gamma_functions

Computes the log-gamma and beta functions using the Lanczos approximation with the Python standard library only.

```python
from simple_beta_and_gamma_functions import beta, gamma, log_beta, log_gamma

log_gamma(5)          # complex(3.1780538303479458, 0.0)
gamma(5)              # complex(24.000000000000004, 0.0)
log_beta(2, 3)        # complex(-2.4849066497880004, 0.0)
beta(2, 3)            # complex(0.08333333333333333, 0.0)
```

All four functions accept real or complex arguments and always return a `complex`.  That decision is deliberate: it means callers never have to special-case the real path, and the reflection formula used for `Re(z) < 0.5` produces a complex result even for real inputs (for example `log_gamma(-0.5)` has an imaginary part of `pi`).

## Why this exists

Most small projects that need a gamma or beta function either pull in a large numerical library or copy a Lanczos routine from a forum post without the reflection formula.  The copied version then returns garbage for negative arguments.  This library is the smallest thing that is still correct on the negative real axis and in the complex plane: one approximation, one reflection step, and a beta function expressed as a difference of log-gammas so that large arguments do not overflow before the division happens.

The trade-off is precision versus size.  The g = 7, n = 9 Lanczos coefficients give roughly 15 significant digits for moderate arguments, which is enough for the typical use cases (statistics, special-function identities, symbolic work) but not enough for high-precision arithmetic.  If you need 50 digits, use `mpmath`; if you need no dependencies, this is the version to copy.

## The awkward edge

The awkward edge is the negative real axis.  `log_gamma` raises `ValueError` at zero and at negative integers, because those are poles and returning `inf` or `nan` would let the error propagate silently into a beta calculation.  Everywhere else on the negative real axis the reflection formula is used, and the result is complex.  A caller who expects a real number from `log_gamma(-0.5)` will be surprised; the README says so, and the tests check it.

## API

- `log_gamma(z)` — natural log of the gamma function at `z`.  Raises `ValueError` if `z` is zero or a negative integer.
- `gamma(z)` — gamma function at `z`, computed as `exp(log_gamma(z))`.  Raises `ValueError` at the same points.
- `log_beta(a, b)` — natural log of the beta function, computed as `log_gamma(a) + log_gamma(b) - log_gamma(a + b)`.  Raises `ValueError` if any of those three log-gammas is undefined.
- `beta(a, b)` — beta function, computed as `exp(log_beta(a, b))`.

## Tests

```
PYTHONPATH=src python -m unittest discover -s tests
```

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

