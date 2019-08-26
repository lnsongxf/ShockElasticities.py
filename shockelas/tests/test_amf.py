"""
A module for testing `amf.py`.

"""

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


def test_iterate(amf, 𝒫_bar, rtol, atol):
    T = 1
    amf.iterate(T)
    𝒫_tilde = amf.𝒫

    test_objs = zip([amf.𝒫_t_bar_path[T], amf.𝒫_t_tilde_path[T]],
                    [𝒫_bar, 𝒫_tilde])

    for 𝒫_test, 𝒫 in test_objs:
        for actual, expected in zip(𝒫_test, 𝒫):
            assert_allclose(actual, expected, rtol=rtol, atol=atol)
