"""Tests for simple_beta_and_gamma_functions.

Run with::

    PYTHONPATH=src python -m unittest discover -s tests
"""

import cmath
import math
import unittest

from simple_beta_and_gamma_functions import (
    beta,
    gamma,
    log_beta,
    log_gamma,
)


class TestLogGamma(unittest.TestCase):
    def assertComplexAlmostEqual(self, got, want, places=9):
        self.assertIsInstance(got, complex)
        self.assertAlmostEqual(got.real, want.real, places=places)
        self.assertAlmostEqual(got.imag, want.imag, places=places)

    def test_real_positive_integers(self):
        # Gamma(n) = (n-1)! for positive integers.
        self.assertComplexAlmostEqual(log_gamma(1), complex(0.0, 0.0))
        self.assertComplexAlmostEqual(log_gamma(2), complex(0.0, 0.0))
        self.assertComplexAlmostEqual(log_gamma(5), complex(math.log(24.0), 0.0))

    def test_real_half_integer(self):
        # Gamma(1/2) = sqrt(pi).
        self.assertComplexAlmostEqual(log_gamma(0.5), complex(math.log(math.sqrt(math.pi)), 0.0))

    def test_real_large(self):
        # Compare against a known value: Gamma(10) = 362880.
        self.assertComplexAlmostEqual(log_gamma(10), complex(math.log(362880.0), 0.0))

    def test_negative_half_integer(self):
        # Gamma(-1/2) = -2 * sqrt(pi).  The reflection formula is exercised here.
        self.assertComplexAlmostEqual(
            log_gamma(-0.5),
            complex(math.log(2.0 * math.sqrt(math.pi)), -math.pi),
        )

    def test_complex_argument(self):
        # Gamma(1 + i) * Gamma(1 - i) = pi / sinh(pi) * pi?  Use the
        # reflection identity directly instead: Gamma(z) * Gamma(1 - z)
        # = pi / sin(pi z).  Comparing logarithms avoids overflow.
        z = complex(0.3, 0.7)
        lhs = log_gamma(z) + log_gamma(1.0 - z)
        rhs = math.log(math.pi) - cmath.log(cmath.sin(math.pi * z))
        self.assertComplexAlmostEqual(lhs, rhs)

    def test_pole_raises(self):
        with self.assertRaises(ValueError):
            log_gamma(0)
        with self.assertRaises(ValueError):
            log_gamma(-3)
        with self.assertRaises(ValueError):
            log_gamma(0.0)

    def test_pole_not_raised_for_negative_real(self):
        # -0.5 is not a pole, so it must not raise.
        value = log_gamma(-0.5)
        self.assertIsInstance(value, complex)


class TestGamma(unittest.TestCase):
    def test_matches_exp_log_gamma(self):
        for z in (0.5, 1.0, 2.5, 5.0, complex(1.0, 1.0)):
            self.assertAlmostEqual(
                abs(gamma(z) - cmath.exp(log_gamma(z))),
                0.0,
                places=12,
            )

    def test_integer_factorial(self):
        self.assertAlmostEqual(abs(gamma(6) - 120.0), 0.0, places=9)

    def test_half_integer(self):
        self.assertAlmostEqual(abs(gamma(0.5) - math.sqrt(math.pi)), 0.0, places=12)

    def test_pole_raises(self):
        with self.assertRaises(ValueError):
            gamma(-1)


class TestLogBeta(unittest.TestCase):
    def assertComplexAlmostEqual(self, got, want, places=9):
        self.assertIsInstance(got, complex)
        self.assertAlmostEqual(got.real, want.real, places=places)
        self.assertAlmostEqual(got.imag, want.imag, places=places)

    def test_symmetric(self):
        self.assertComplexAlmostEqual(log_beta(2, 3), log_beta(3, 2))

    def test_known_value(self):
        # B(2, 3) = 1/12, so log B(2, 3) = -log(12).
        self.assertComplexAlmostEqual(log_beta(2, 3), complex(-math.log(12.0), 0.0))

    def test_relation_to_gamma(self):
        a, b = 1.5, 2.5
        lhs = log_beta(a, b)
        rhs = log_gamma(a) + log_gamma(b) - log_gamma(a + b)
        self.assertComplexAlmostEqual(lhs, rhs)

    def test_complex_arguments(self):
        a = complex(0.5, 0.25)
        b = complex(0.75, -0.125)
        lhs = log_beta(a, b)
        rhs = log_gamma(a) + log_gamma(b) - log_gamma(a + b)
        self.assertComplexAlmostEqual(lhs, rhs)

    def test_pole_raises(self):
        with self.assertRaises(ValueError):
            log_beta(0, 1)
        with self.assertRaises(ValueError):
            log_beta(1, 0)


class TestBeta(unittest.TestCase):
    def test_matches_exp_log_beta(self):
        for a, b in ((2, 3), (0.5, 0.5), (1.5, 2.5)):
            self.assertAlmostEqual(
                abs(beta(a, b) - cmath.exp(log_beta(a, b))),
                0.0,
                places=12,
            )

    def test_known_value(self):
        self.assertAlmostEqual(abs(beta(2, 3) - 1.0 / 12.0), 0.0, places=12)

    def test_symmetric(self):
        self.assertAlmostEqual(abs(beta(2, 3) - beta(3, 2)), 0.0, places=12)

    def test_pole_raises(self):
        with self.assertRaises(ValueError):
            beta(-1, 2)


if __name__ == "__main__":
    unittest.main()
