import numpy as np
from scipy.optimize import minimize
from Model_Interface import ModelInterface

"""
File: Model_RescorlaWagner.py
This module implements the Rescorla-Wagner model as a concrete implementation of the abstract base class (ABC) defined in Model_Interface.py.
The Rescorla-Wagner model is a classic reinforcement learning model that updates action values (Q-values) based on prediction errors.
It uses a softmax function to convert Q-values into action probabilities and has two free parameters: the learning rate (alpha) and the inverse temperature (beta).
The model can be fitted to behavioral data (action and reward sequences) using maximum likelihood estimation (MLE), and it can also simulate behavior in a given environment.
See the main Hack notebook for a more detailed explanation of the model and its parameters.

Notes for adaptation / extensions of this class:
- the model interface requires you to implement fitting and probability generation methods, but you can add any additional methods or attributes you want (e.g., for visualization, logging, etc.)
- adding a method for returning latent states is useful for visualization, debugging, and interpretability
- providing the number of parameters will help assess / compare model complexity
- given that `get_action_probabilities` is provided, the Model_Interface automatically provides a `simulate` method that can be used to generate synthetic data from the model in a given environment. This is useful for testing and validation.
"""

class RescorlaWagnerModel(ModelInterface):
    def __init__(self, task_properties: dict):
        super().__init__(task_properties) # This model only needs to know the nr of actions
        self.alpha = 0.5 # Learning rate (initialized to a default; will be fitted either to data or during simulation)
        self.beta = 1.0 # Inverse temperature for softmax
        self.q_history = None  # Stores latent states (Q-values)

    def get_num_parameters(self) -> int:
        return 2  # alpha and beta, q_history is only for visualization and not a free parameter

    def get_action_probabilities(self, actions: np.ndarray, rewards: np.ndarray, alpha=None, beta=None) -> np.ndarray:
        """Runs the trial-by-trial Rescorla-Wagner loop to generate choice probabilities."""
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta

        n_trials = len(actions)
        action_probs = np.zeros((n_trials, self.n_actions))
        # initialize Q0 at the session's mean reward so the scale matches the actual rewards (not just [0,1])
        initial_value = float(np.mean(rewards)) if len(rewards) > 0 else 0.5
        q_values = np.ones(self.n_actions) * initial_value

        # tracking Q-values over time for get_latent_states()
        self.q_history = np.zeros((n_trials, self.n_actions))

        for t in range(n_trials):
            # 1. Action Selection: Softmax probabilities
            exp_q = np.exp(beta * (q_values - np.max(q_values))) # subtract max for numerical stability, changes nothing about the softmax output
            probs = exp_q / np.sum(exp_q)
            action_probs[t, :] = probs
            self.q_history[t, :] = q_values.copy()

            # 2. Value Updating (Prediction Error)
            chosen_action = int(actions[t])
            delta_t = rewards[t] - q_values[chosen_action]
            q_values[chosen_action] += alpha * delta_t

        return action_probs

    def fit(self, actions_list: list, rewards_list: list):
        """Fits alpha and beta to a population of subjects by minimizing summed NLL.
        actions_list and rewards_list are lists of 1D arrays, one session per subject.
        """
        # helper to compute negative log-likelihood for a single subject, we'll use this to compute the total NLL across all subjects
        def compute_nll(probs, acts):
            epsilon = 1e-8
            chosen_probs = probs[np.arange(len(acts)), acts]
            return -np.sum(np.log(chosen_probs + epsilon))

        # this is will be fed to the optimizer to be minimized
        def objective_function(params: list):
            alpha_guess, beta_guess = params
            total_nll = 0
            # iterate through all subjects in the list
            for acts, rews in zip(actions_list, rewards_list):
                probs = self.get_action_probabilities(acts, rews, alpha_guess, beta_guess)
                total_nll += compute_nll(probs, acts)
            return total_nll

        initial_guess = [0.5, 1.0]
        bounds = [(0.0, 1.0), (1e-6, 50.0)] # alpha is bounded between 0 and 1, beta is bounded to be positive (but not zero)

        # `minimize` computes gradient (for direction of steepest descent) and hessian (for deciding step size); is a quasi-Newton method suitable for box-constrained ('B') optimization
        # feel free to experiment with other optimizers (e.g., Nelder-Mead, Powell, etc.) if you want to see how they perform on this problem
        # OR you can solve for the analytical gradients of alpha and beta and feed them to the optimizer if you're feeling fancy
        result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')

        if result.success:
            self.alpha, self.beta = result.x
            self.is_fitted = True
            print(f"Fit successful! Alpha: {self.alpha:.3f}, Beta: {self.beta:.3f}; NLL: {result.fun:.3f}")
        else:
            print("Optimization failed:", result.message)

        return self

    def get_latent_states(self) -> np.ndarray:
        """Returns the internal Q-values generated during the last probability pass."""
        return self.q_history

    def get_model_description(self) -> dict:
        """Overrides base method to inject specific parameter values."""
        base_desc = super().get_model_description()
        base_desc["description"] = "Standard Rescorla-Wagner model with Softmax choice."
        if self.is_fitted:
            base_desc["parameters"] = f"Alpha: {self.alpha:.3f}, Beta: {self.beta:.3f}"
        else:
            base_desc["parameters"] = "Unfitted"

        return base_desc
