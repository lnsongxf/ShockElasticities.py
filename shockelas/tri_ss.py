"""
A module for representing triangular state space vector systems.

"""

class TriSS:
    def __init__(self, Θ_10, Θ_11, Λ_10, Θ_20, Θ_21, Θ_22, Θ_23, Λ_20, Λ_21,
                 Λ_22):
        self.Θ_10 = Θ_10
        self.Θ_11 = Θ_11
        self.Λ_10 = Λ_10
        self.Θ_20 = Θ_20
        self.Θ_21 = Θ_21
        self.Θ_22 = Θ_22
        self.Θ_23 = Θ_23
        self.Λ_20 = Λ_20
        self.Λ_21 = Λ_21
        self.Λ_22 = Λ_22


def map_perturbed_model_to_tri_ss(perturbed_model_params):
    """
    Maps parameters from the perburbed model into the triangular system.

    """

    tri_ss_params = {
        'Θ_10': perturbed_model_params['ψ_q'],
        'Θ_11': perturbed_model_params['ψ_x'],
        'Λ_10': perturbed_model_params['ψ_w'],
        'Θ_20': perturbed_model_params['ψ_qq'],
        'Θ_21': 2 * perturbed_model_params['ψ_xq'],
        'Θ_22': perturbed_model_params['ψ_x'],
        'Θ_23': perturbed_model_params['ψ_xx'],
        'Λ_20': 2 * perturbed_model_params['ψ_wq'],
        'Λ_21': 2 * perturbed_model_params['ψ_xw'],
        'Λ_22': perturbed_model_params['ψ_ww']
        }

    tri_ss = TriSS(*tri_ss_params.values())

    return tri_ss
