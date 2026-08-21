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
    def fit(self, actions_list: list, rewards_list: list, states_list: list = None, n_actions_list: list = None):
        """
        Fits the model to the provided action and reward sequences.
        Accepts lists of 1D arrays (each element is one subject's session) to support
        both SciPy iteration (subject-by-subject) and PyTorch batching.

        Optional state-aware inputs (ignored by models that do not use them):
        - states_list: list of per-session 1D integer arrays with the state index at
          each trial (convention: integer one-hot indices in 0..max_n_states-1;
          for Azulejos these are the global state labels, stored by DataManager in
          the 'states_idx' column).
          The GRU uses states to build its input at trial t: [a_{t-1}, available
          actions, s_t, s_{t-1}, r_{t-1}].
        - n_actions_list: list of ints, one per session, for models with a fixed
          output space shared across tasks of different sizes (e.g. GRU outputs 6
          arms and masks unavailable ones). Defaults to the model's own n_actions.

        Returns:
            self
        """
        pass

    @abstractmethod
    def get_action_probabilities(self, actions: np.ndarray, rewards: np.ndarray,
                                 states: np.ndarray = None, n_actions: int = None) -> np.ndarray:
        """
        Returns trial-by-trial action probabilities from history.

        IMPORTANT temporal convention:
        - Probability at trial t must be computed from history up to t-1.
        - Each row represents P(A_t | past actions, past rewards).
        - This keeps likelihood evaluation and simulation aligned across
          classical and recurrent models.

        Optional inputs (state-aware models only):
        - states: 1D integer array with the state index at each trial
          (global one-hot indices, 0..max_n_states-1). For the GRU, trial t uses
          s_t as part of its input, so states must cover up to the current trial.
        - n_actions: number of available actions for this sequence, used to mask
          the model's output when it uses a fixed output space shared across tasks.

        Returns:
            2D numpy array of shape (N_trials, N_actions) where each row sums to 1.
        """
        pass

    def simulate(self, environment, n_trials: int, states: np.ndarray = None, n_actions: int = None) -> tuple:
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

        Optional state-aware inputs:
        - states: 1D integer array of state indices, one per trial. If provided, it
          is passed to get_action_probabilities() (requires a state-aware model,
          e.g. the GRU); otherwise the model is simulated without state input.
        - n_actions: number of available actions for this environment, passed
          through for fixed-output models that mask unavailable actions.
        """
        # check if model is fitted; if not, raise a warning but still allow simulation
        if not self.is_fitted:
            print(f"Warning: Model {self.__class__.__name__} is not fitted. Simulating with default parameters.")

        environment.reset()
        simulated_actions = np.zeros(n_trials, dtype=int)
        simulated_rewards = np.zeros(n_trials, dtype=float)
        n_actions = self.n_actions if n_actions is None else int(n_actions)

        # empty histories
        history_actions = np.array([], dtype=int)
        history_rewards = np.array([], dtype=float)
        history_states = np.array([], dtype=int)

        for t in range(n_trials):
            # get probabilities for current step based on history up to t-1
            # (append a dummy step to get the prediction for trial t)
            temp_actions = np.append(history_actions, 0)
            temp_rewards = np.append(history_rewards, 0)
            if states is not None:
                # state at trial t is part of the input for prediction at trial t
                temp_states = np.append(history_states, int(states[t]))
                probs = self.get_action_probabilities(temp_actions, temp_rewards, temp_states, n_actions)[-1]
            else:
                probs = self.get_action_probabilities(temp_actions, temp_rewards)[-1]

            # model makes a choice based on its internal probabilities
            action = np.random.choice(n_actions, p=probs[:n_actions])

            # environment provides feedback
            reward = environment.step(action)

            # save info and update history
            simulated_actions[t] = action
            simulated_rewards[t] = reward
            history_actions = np.append(history_actions, action)
            history_rewards = np.append(history_rewards, reward)
            if states is not None:
                history_states = np.append(history_states, int(states[t]))

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

    def get_latent_states_for_sequence(self, actions: np.ndarray, rewards: np.ndarray,
                                       states: np.ndarray = None, n_actions: int = None):
        """
        Optional convenience helper for extracting latents on a provided sequence.

        Default behavior runs get_action_probabilities() on the sequence and returns
        get_latent_states(). Models may override this method for efficiency.
        """
        _ = self.get_action_probabilities(actions, rewards, states, n_actions)
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
