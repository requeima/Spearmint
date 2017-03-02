import numpy          as np
import scipy.stats    as sps
from spearmint.acquisition_functions.abstract_acquisition_function import AbstractAcquisitionFunction
from spearmint.acquisition_functions.constraints_helper_functions import total_constraint_confidence
from spearmint.acquisition_functions.predictive_entropy_search import sample_gp_with_random_features

# in PES, NUM_RANDOM_FEATURES is hardcoded to 1000
NUM_RANDOM_FEATURES = 1000


class TS(AbstractAcquisitionFunction):
    def __init__(self, num_dims, verbose=True, input_space=None, grid=None, opt = None):
        self.cashed_thompson_sample = None
        self.has_gradients = False

    def acquisition(self, objective_model_dict, constraint_models_dict, cand, current_best, compute_grad, tasks=None):
        obj_model = objective_model_dict.values()[0]

        if compute_grad:
            self.has_gradients = True

        if not self.cashed_thompson_sample:
            self.cashed_thompson_sample = sample_gp_with_random_features(obj_model, NUM_RANDOM_FEATURES)

        objective = -self.cashed_thompson_sample(cand, gradient=False)
        if compute_grad:
            gradient = self.cashed_thompson_sample(cand, gradient=True)
            return objective, gradient
        else:
            return objective