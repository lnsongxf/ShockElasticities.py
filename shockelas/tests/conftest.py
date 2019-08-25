"""
A module containing useful pytest fixtures.

"""

import numpy as np
import pytest
from shockelas import TriSS, Amf


# Parameters
κ = 1e-4

# X_t: 2x1, W_t: 3x1
ψ_q = κ * np.array([[1.], [2.]])
ψ_x = κ * np.array([[3., 4.],
                    [5., 6.]])
ψ_w = κ * np.array([[7., 8., 9.],
                    [10., 11., 12.]])

ψ_qq = κ * np.array([[13.],
                     [14.]])

ψ_xq = κ * np.array([[15., 16.],
                     [17., 18.]])

ψ_xx = κ * np.array([[19., 20., 21., 22.],
                     [23., 24., 25., 26.]])

ψ_xw = κ * np.array([[27., 28., 29., 30., 31., 32.],
                     [33., 34., 35., 36., 37., 38.]])

ψ_wq = κ * np.array([[39., 40., 41.],
                     [42., 43., 44.]])

ψ_ww = κ * np.array([[45., 46., 47., 48., 49., 50., 51., 52., 53.],
                     [54., 55., 56., 57., 58., 59., 60., 61., 62.]])

# Y_t: 1x1
Γ_0 = κ * np.array([[63.]])

Γ_1 = κ * np.array([[64., 65.]])

Γ_2 = κ * np.array([[66., 67.]])

Γ_3 = κ * np.array([[68., 69., 70., 71.]])

Ψ_0 = κ * np.array([[72., 73., 74.]])

Ψ_1 = κ * np.array([[75., 76., 77., 78., 79., 80.]])

Ψ_2 = κ * np.array([[81., 82., 83., 84., 85., 86., 87., 88., 89.]])

# Fixtures

@pytest.fixture(scope="session")
def rtol():
    "Relative tolerance level for tests."

    rtol = 1e-8
    return rtol


@pytest.fixture(scope="session")
def atol():
    "Absolute tolerance level for tests."

    atol = 1e-8
    return atol


@pytest.fixture(scope="session")
def perturbed_model_params():
    perturbed_model_params = {
        'ψ_q': ψ_q,
        'ψ_x': ψ_x,
        'ψ_w' : ψ_w,
        'ψ_qq' : ψ_qq,
        'ψ_xq' : ψ_xq,
        'ψ_xx' : ψ_xx,
        'ψ_xw' : ψ_xw,
        'ψ_wq' : ψ_wq,
        'ψ_ww' : ψ_ww
    }

    return perturbed_model_params


@pytest.fixture(scope="session")
def tri_ss():
    tri_params = {
        'Θ_10': ψ_q,
        'Θ_11': ψ_x,
        'Λ_10': ψ_w,
        'Θ_20': ψ_qq,
        'Θ_21': 2 * ψ_xq,
        'Θ_22': ψ_x,
        'Θ_23': ψ_xx,
        'Λ_20': 2 * ψ_wq,
        'Λ_21': 2 * ψ_xw,
        'Λ_22': ψ_ww
    }

    tri_ss = TriSS(*tri_params.values())

    return tri_ss


@pytest.fixture(scope="session")
def amf(tri_ss):
    𝒫 = (Γ_0, Γ_1, Γ_2, Γ_3, Ψ_0, Ψ_1, Ψ_2)

    amf = Amf(𝒫, tri_ss)

    return amf


@pytest.fixture(scope="session")
def 𝒫_bar():
    # Ɛ(𝒫)
    Γ_0_bar = np.array([[0.03255799]])
    Γ_1_bar = np.array([[0.00657541, 0.00668233]])
    Γ_3_bar = np.array([[0.00689131, 0.00699491, 0.00709491, 0.00719866]])

    𝒫_bar = (Γ_0_bar, Γ_1_bar, Γ_2, Γ_3_bar, np.array([[0.]]),
              np.array([[0.]]), np.array([[0.]]))

    return 𝒫_bar


@pytest.fixture(scope="session")
def 𝒫_tilde():
    Γ_0_tilde = np.array([[0.03257794]])
    Γ_1_tilde = np.array([[4.78971915e-05, 5.18838140e-05]])
    Γ_2_tilde = np.array([[5.33e-06, 6.66e-06]])
    Γ_3_tilde = np.array([[2.79545334e-05, 2.92856646e-05, 3.06156626e-05,
                           3.19470757e-05]])
    Ψ_0_tilde = np.array([[0.00011905, 0.00012304, 0.00012703]])
    Ψ_1_tilde = np.array([[7.98792514e-05, 8.25415119e-05, 8.52037725e-05,
                           8.78640512e-05, 9.05268753e-05,
                           9.31896994e-05]])
    Ψ_2_tilde = np.array([[6.59004383e-05, 6.72328397e-05, 6.85652411e-05,
                           6.98928367e-05, 7.12255199e-05, 7.25582030e-05,
                           7.38852351e-05, 7.52182000e-05,
                           7.65511650e-05]])

    𝒫_tilde = (Γ_0_tilde, Γ_1_tilde, Γ_2_tilde, Γ_3_tilde, Ψ_0_tilde,
                Ψ_1_tilde, Ψ_2_tilde)

    return 𝒫_tilde
