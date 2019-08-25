"""
A module for testing `tri_ss.py`.

"""

import numpy as np
from numpy.testing import assert_allclose
from shockelas import map_perturbed_model_to_tri_ss


def test_map_perturbed_model_to_tri_ss(perturbed_model_params, tri_ss, rtol,
									   atol):
    tri_ss_test = map_perturbed_model_to_tri_ss(perturbed_model_params)

    attrs = ['Θ_10', 'Θ_11', 'Λ_10', 'Θ_20', 'Θ_21', 'Θ_22', 'Θ_23','Λ_20',
             'Λ_21', 'Λ_22']

    for attr in attrs:
        assert_allclose(tri_ss_test.__getattribute__(attr),
                        tri_ss.__getattribute__(attr), rtol=rtol, atol=atol)
