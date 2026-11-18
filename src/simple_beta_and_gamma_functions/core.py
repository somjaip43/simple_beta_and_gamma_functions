"""Lanczos approximation for the gamma function and the beta function.

The Lanczos approximation gives the gamma function as

    Gamma(z) = sqrt(2*pi) * (z + g - 0.5)**(z - 0.5)
               * exp(-(z + g - 0.5)) * A_g(z)

where A_g(z) is a rational series.  We use the standard g = 7, n = 9
coefficients (the same ones used by Numerical Recipes and the GNU
Scientific Library).  Working in the log domain avoids overflow for
moderate z and makes the beta function a one-line difference.

All functions accept real or complex Python numbers.  For complex
arguments the branch cut of log-gamma runs along the negative real axis,
as is conventional.

References:
    Lanczos, C. (1964). A precision approximation of the gamma function.
    SIAM Journal on Numerical Analysis, 1(1), 86-96.
"""

from __future__ import annotations

import cmath
import math
from typing import Union

Number = Union[int, float, complex]

# g = 7, n = 9 Lanczos coefficients.
_G = 7.0
_COEFFS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _lanczos_sum(z: complex) -> complex:
    """Evaluate the Lanczos rational series A_g(z) for a complex z.

    The series is written as a nested sum so that each coefficient is
    multiplied by exactly one reciprocal power of (z + i).  This keeps
    the numerical error close to machine precision for the small n we
    use here.
    """
    # Start with the constant term and add the tail from the smallest
    # coefficient upward; this ordering minimises round-off when the
    # higher terms are small relative to the leading one.
    acc = complex(_COEFFS[0])
    for i in range(1, len(_COEFFS)):
        acc += _COEFFS[i] / (z + i)
    return acc


def log_gamma(z: Number) -> complex:
    """Return the natural logarithm of the gamma function at ``z``.

    The argument may be real or complex.  For real positive ``z`` the
    result is real (but returned as a ``complex`` so that callers do not
    have to branch on the type).  A pole at zero or a negative integer
    raises ``ValueError`` rather than returning an infinite value,
    because silently returning ``inf`` hides the singularity from
    callers who would then compute nonsense downstream.

    The reflection formula is used for ``Re(z) < 0.5`` so that the
    Lanczos series, which converges well only in the right half-plane,
    is never evaluated in a region where it loses accuracy.
    """
    zc = complex(z)

    # Poles at non-positive integers.
    if zc.imag == 0.0 and zc.real <= 0.0 and zc.real == math.floor(zc.real):
        raise ValueError("log_gamma is undefined at non-positive integers")

    if zc.real < 0.5:
        # Reflection: Gamma(z) * Gamma(1 - z) = pi / sin(pi * z).
        # log Gamma(z) = log(pi) - log(sin(pi * z)) - log Gamma(1 - z).
        # sin(pi * z) can be zero only at integers, which we excluded above.
        return (
            math.log(math.pi)
            - cmath.log(cmath.sin(math.pi * zc))
            - log_gamma(1.0 - zc)
        )

    shifted = zc - 1.0
    series = _lanczos_sum(shifted)
    t = shifted + _G + 0.5
    # log Gamma(z) = 0.5 * log(2*pi) + (z - 0.5) * log(t) - t + log(A_g(z)).
    return (
        math.log(_SQRT_2PI)
        + (zc - 0.5) * cmath.log(t)
        - t
        + cmath.log(series)
    )


def gamma(z: Number) -> complex:
    """Return the gamma function at ``z``.

    This is a thin wrapper around :func:`log_gamma`; it exists so that
    callers who want the value rather than its logarithm do not have to
    write ``cmath.exp(log_gamma(z))`` themselves.  The result is always
    a ``complex`` for the same reason as in :func:`log_gamma`.
    """
    return cmath.exp(log_gamma(z))


def log_beta(a: Number, b: Number) -> complex:
    """Return the natural logarithm of the beta function B(a, b).

    B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b), so
    log B(a, b) = log Gamma(a) + log Gamma(b) - log Gamma(a + b).
    The arguments may be real or complex.  Poles of the individual
    gamma factors propagate as ``ValueError`` from :func:`log_gamma`.
    """
    return log_gamma(a) + log_gamma(b) - log_gamma(complex(a) + complex(b))


def beta(a: Number, b: Number) -> complex:
    """Return the beta function B(a, b).

    Computed as ``exp(log_beta(a, b))`` to avoid the intermediate
    overflow that a direct product of gamma values can hit for large
    arguments.  The result is always a ``complex``.
    """
    return cmath.exp(log_beta(a, b))
