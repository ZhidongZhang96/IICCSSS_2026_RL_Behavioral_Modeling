from abc import ABC, abstractmethod
import numpy as np

"""
File: Model_Interface.py
Shared model contract for workshop models.

Any model used in the notebook should provide:
1) fit on action/reward sequences,
2) trial-wise action probabilities,
3) optional latent state access.

The base class also provides a generic simulate() method that samples actions from
the model's probability outputs and receives rewards from an environment.
"""

class ModelInterface(ABC):
    def __init__(self, task_properties: dict):
        """
        Initializes the model dynamically based on the task environment.
        task_properties should contain keys like 'n_actions'.
        """
        self.n_actions = task_properties.get('n_actions', 2)
        self.is_fitted = False

    @abstractmethod
    def fit(self, actions_list: list, rewards_list: list):
        """
        Fits the model to the provided action and reward sequences.
        Accepts lists of 1D arrays (each element is one subject's session) to support
        both SciPy iteration (subject-by-subject) and PyTorch batching.

        Returns:
            self
        """
        pass

    @abstractmethod
    def get_action_probabilities(self, actions: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Returns trial-by-trial action probabilities from history.

        IMPORTANT temporal convention:
        - Probability at trial t must be computed from history up to t-1.
        - Each row represents P(A_t | past actions, past rewards).
        - This keeps likelihood evaluation and simulation aligned across
          classical and recurrent models.

        Returns:
            2D numpy array of shape (N_trials, N_actions) where each row sums to 1.
        """
        pass

    def simulate(self, environment, n_trials: int) -> tuple:
        """
        The model acts autonomously in a simulated environment.

        This default implementation is intentionally generic and should work for any
        model implementing get_action_probabilities(). At each trial it:
        1) queries probabilities from current history,
        2) samples an action from that distribution,
        3) obtains reward feedback from the environment,
        4) appends action/reward to history.

        Models can override this method, but we recommend keeping this shared logic
        unless model-specific simulation behavior is required.
        """
        # check if model is fitted; if not, raise a warning but still allow simulation
        if not self.is_fitted:
            print(f"Warning: Model {self.__class__.__name__} is not fitted. Simulating with default parameters.")

        environment.reset()
        simulated_actions = np.zeros(n_trials, dtype=int)
        simulated_rewards = np.zeros(n_trials, dtype=float)

        # empty histories
        history_actions = np.array([], dtype=int)
        history_rewards = np.array([], dtype=float)

        for t in range(n_trials):
            # get probabilities for current step based on history up to t-1
            # (append a dummy step to get the prediction for trial t)
            temp_actions = np.append(history_actions, 0)
            temp_rewards = np.append(history_rewards, 0)

            probs = self.get_action_probabilities(temp_actions, temp_rewards)[-1]

            # model makes a choice based on its internal probabilities
            action = np.random.choice(self.n_actions, p=probs)

            # environment provides feedback
            reward = environment.step(action)

            # save info and update history
            simulated_actions[t] = action
            simulated_rewards[t] = reward
            history_actions = np.append(history_actions, action)
            history_rewards = np.append(history_rewards, reward)

        return simulated_actions, simulated_rewards

    def get_latent_states(self):
        """
        Returns internal variables for visualization (e.g., Q-values or RNN hidden states).
        Models should override this if they have trackable latent dynamics.
        Ideally, this should return a 2D array of shape (N_trials, N_latent_features) for easy plotting,
        but beware of large latent spaces (e.g., RNNs with hundreds of hidden units).
        In such cases, consider returning a PCA-reduced version of the latent states.

        By convention, this returns latent states from the most recent probability pass.
        """
        return None

    def get_latent_states_for_sequence(self, actions: np.ndarray, rewards: np.ndarray):
        """
        Optional convenience helper for extracting latents on a provided sequence.

        Default behavior runs get_action_probabilities() on the sequence and returns
        get_latent_states(). Models may override this method for efficiency.
        """
        _ = self.get_action_probabilities(actions, rewards)
        return self.get_latent_states()

    def get_num_parameters(self) -> int:
        """
        Returns the number of free parameters. Required for AIC/BIC penalization.
        Models with fitted parameters (like $\alpha$ and $\beta$) must override this.
        """
        return 0

    # optional but might be useful to stay organized when using different models
    def get_model_description(self) -> dict:
        """
        Returns a description of the model.
        Models with fitted parameters should override this to provide meaningful descriptions.
        """
        latent = self.get_latent_states()
        if hasattr(latent, "shape"):
            latent_shape = tuple(latent.shape)
        else:
            latent_shape = "N/A"

        description = {
            "name": self.__class__.__name__,
            "fitted_status": "Fitted" if self.is_fitted else "Unfitted",
            "description": "N/A",
            "n_parameters": self.get_num_parameters(),
            "latent_shape": latent_shape,
            }

        return description
