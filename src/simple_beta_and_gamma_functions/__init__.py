"""Log-gamma and beta functions built on the Lanczos approximation.

Exports:
    log_gamma(z)  -- natural log of the gamma function for complex z
    gamma(z)      -- gamma function for complex z, via exp(log_gamma(z))
    log_beta(a, b) -- natural log of the beta function for complex a, b
    beta(a, b)     -- beta function for complex a, b, via exp(log_beta(a, b))
"""

from .core import beta, gamma, log_beta, log_gamma

__all__ = ["log_gamma", "gamma", "log_beta", "beta"]
