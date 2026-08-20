import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax

from Model_Interface import ModelInterface

"""
File: Model_GRU.py
Replicates the RNN fitting method used in Sarah Master's IICCSSS workshop notebook
(dm-haiku GRU trained with the Adam optimizer over fully-batched session tensors),
re-implemented here directly in raw JAX + optax (no haiku dependency) so it follows
the same ModelInterface contract as RescorlaWagnerModel and SimpleRNNModel.

How this differs from the plain Elman-style SimpleRNNModel (Model_RNN.py):
- Recurrent cell: a gated recurrent unit (reset/update gates + candidate state)
  instead of a single tanh layer.
- Optimizer: Adam (via optax), instead of plain gradient descent.
- Batching: every training session is padded to equal length and processed in one
  vectorized forward/backward pass (a single lax.scan over time, batched across all
  sessions at once, mirroring hk.dynamic_unroll on a stacked tensor), instead of
  looping over sessions in Python. Padded trials are excluded via a boolean mask.

Note: the workshop's original GRU also conditioned on task state (multi-state
Azulejos tasks) and an available-actions mask. Those aren't part of this notebook's
ModelInterface (fit()/get_action_probabilities() only take action/reward history),
so this class uses the same previous-action + previous-reward features as
SimpleRNNModel to stay directly comparable within this notebook.
"""


class SimpleGRUModel(ModelInterface):
    def __init__(
        self,
        task_properties: dict,
        hidden_size: int = 16,
        learning_rate: float = 0.01,
        n_epochs: int = 50,
        seed: int = 0,
    ):
        super().__init__(task_properties)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self.input_size = int(self.n_actions + 1)  # previous action one-hot + previous reward

        self.params = self._initialize_parameters()
        self.loss_history = []
        self.test_loss_history = []
        self.hidden_history = None

    def _initialize_parameters(self):
        rng = np.random.default_rng(self.seed)
        scale = 0.1
        hidden, inp, out = self.hidden_size, self.input_size, self.n_actions

        def w(shape):
            return jnp.array(rng.normal(scale=scale, size=shape))

        return {
            "Wxz": w((hidden, inp)), "Whz": w((hidden, hidden)), "bz": jnp.zeros((hidden,)),
            "Wxr": w((hidden, inp)), "Whr": w((hidden, hidden)), "br": jnp.zeros((hidden,)),
            "Wxh": w((hidden, inp)), "Whh": w((hidden, hidden)), "bh": jnp.zeros((hidden,)),
            "Wy": w((out, hidden)), "by": jnp.zeros((out,)),
        }

    def _build_batch(self, actions_list: list, rewards_list: list):
        """Pads variable-length sessions into one (session, time, feature) tensor plus a valid-trial mask."""
        max_len = max(len(np.asarray(a, dtype=int)) for a in actions_list)
        n_sessions = len(actions_list)

        X = np.zeros((n_sessions, max_len, self.input_size), dtype=float)
        Y = np.zeros((n_sessions, max_len), dtype=int)
        mask = np.zeros((n_sessions, max_len), dtype=bool)

        for i, (actions, rewards) in enumerate(zip(actions_list, rewards_list)):
            actions = np.asarray(actions, dtype=int)
            rewards = np.asarray(rewards, dtype=float)
            n_trials = len(actions)

            for t in range(1, n_trials):
                prev_action = int(actions[t - 1])
                if 0 <= prev_action < self.n_actions:
                    X[i, t, prev_action] = 1.0
                X[i, t, -1] = rewards[t - 1]

            Y[i, :n_trials] = actions
            mask[i, :n_trials] = True

        return jnp.array(X), jnp.array(Y), jnp.array(mask)

    def _forward_logits(self, params, x_batch: jnp.ndarray):
        """Vectorized GRU forward pass: one lax.scan over time, batched across all sessions."""
        x_time_major = jnp.transpose(x_batch, (1, 0, 2))  # (time, batch, input)

        def step_fn(h_prev, x_t):
            z = jax.nn.sigmoid(x_t @ params["Wxz"].T + h_prev @ params["Whz"].T + params["bz"])
            r = jax.nn.sigmoid(x_t @ params["Wxr"].T + h_prev @ params["Whr"].T + params["br"])
            h_tilde = jnp.tanh(x_t @ params["Wxh"].T + (r * h_prev) @ params["Whh"].T + params["bh"])
            h_t = (1 - z) * h_prev + z * h_tilde
            logits_t = h_t @ params["Wy"].T + params["by"]
            return h_t, (logits_t, h_t)

        batch_size = x_batch.shape[0]
        h0 = jnp.zeros((batch_size, self.hidden_size))
        _, (logits_seq, hidden_seq) = lax.scan(step_fn, h0, x_time_major)
        logits = jnp.transpose(logits_seq, (1, 0, 2))  # (batch, time, n_actions)
        hidden = jnp.transpose(hidden_seq, (1, 0, 2))  # (batch, time, hidden)
        return logits, hidden

    def _masked_nll(self, params, X, Y, mask):
        logits, _ = self._forward_logits(params, X)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(Y, num_classes=self.n_actions)
        trial_log_liks = jnp.sum(one_hot * log_probs, axis=-1)
        masked_log_liks = jnp.where(mask, trial_log_liks, 0.0)
        total_nll = -jnp.sum(masked_log_liks)
        n_valid = jnp.maximum(jnp.sum(mask), 1)
        return total_nll / n_valid

    def fit(self, actions_list: list, rewards_list: list, val_actions_list: list = None, val_rewards_list: list = None):
        """Fit GRU parameters with Adam over the full (batched) training set per epoch.

        If val_actions_list/val_rewards_list are given, the held-out NLL is also
        tracked every epoch (mirrors the train/test loss curve in Sarah's workshop notebook).
        """
        if len(actions_list) == 0:
            raise ValueError("actions_list is empty; cannot fit model.")

        X, Y, mask = self._build_batch(actions_list, rewards_list)

        track_val = val_actions_list is not None and len(val_actions_list) > 0
        if track_val:
            X_val, Y_val, mask_val = self._build_batch(val_actions_list, val_rewards_list)

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params)

        def objective(params, X, Y, mask):
            return self._masked_nll(params, X, Y, mask)

        value_and_grad_fn = jax.value_and_grad(objective)
        eval_loss_fn = jax.jit(objective)

        @jax.jit
        def train_step(params, opt_state, X, Y, mask):
            loss_value, grads = value_and_grad_fn(params, X, Y, mask)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        self.loss_history = []
        self.test_loss_history = []
        params = self.params
        for _ in range(self.n_epochs):
            if track_val:
                self.test_loss_history.append(float(eval_loss_fn(params, X_val, Y_val, mask_val)))
            params, opt_state, loss_value = train_step(params, opt_state, X, Y, mask)
            self.loss_history.append(float(loss_value))

        self.params = params
        self.is_fitted = True

        final_loss = self.loss_history[-1] if self.loss_history else float(objective(params, X, Y, mask))
        print(f"GRU fit complete. Mean NLL/trial: {final_loss:.4f}")
        return self

    def get_action_probabilities(self, actions: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Return P(A_t | history up to t-1) for all trials in a sequence."""
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        if len(actions) == 0:
            self.hidden_history = np.zeros((0, self.hidden_size), dtype=float)
            return np.zeros((0, self.n_actions), dtype=float)

        X, _, _ = self._build_batch([actions], [rewards])
        logits, hidden = self._forward_logits(self.params, X)
        probs = jax.nn.softmax(logits[0], axis=-1)

        self.hidden_history = np.asarray(hidden[0])
        return np.asarray(probs)

    def get_latent_states(self):
        return self.hidden_history

    def get_num_parameters(self) -> int:
        total = 0
        for value in self.params.values():
            total += int(np.prod(value.shape))
        return total

    def get_model_description(self) -> dict:
        base_desc = super().get_model_description()
        base_desc["description"] = (
            "GRU policy model trained with Adam over fully-batched session tensors, "
            "replicating the fitting method from Sarah Master's IICCSSS workshop notebook."
        )
        base_desc["hidden_size"] = f"{self.hidden_size}"
        base_desc["learning_rate"] = f"{self.learning_rate}"
        base_desc["epochs"] = f"{self.n_epochs}"
        return base_desc
