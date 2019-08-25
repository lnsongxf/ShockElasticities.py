"""
A module for testing `amf.py`.

"""


import numpy as np
from numpy.testing import assert_allclose


def test_Ɛ_bar(amf, 𝒫_bar, rtol, atol):
    𝒫 = amf.𝒫
    𝒫_bar_test = amf.Ɛ_bar(𝒫)

    for actual, expected in zip(𝒫_bar_test, 𝒫_bar):
        assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_Ɛ_tilde(amf, 𝒫_bar, 𝒫_tilde, rtol, atol):
    𝒫_tilde_test = amf.Ɛ_tilde(𝒫_bar)

    for actual, expected in zip(𝒫_tilde_test, 𝒫_tilde):
        assert_allclose(actual, expected, rtol=rtol, atol=atol)
