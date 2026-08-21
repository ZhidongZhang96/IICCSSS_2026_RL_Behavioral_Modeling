import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax

from Model_Interface import ModelInterface

"""
File: Model_GRU.py
GRU policy model that replicates the fitting method from Sarah Master's IICCSSS
workshop notebook (Adam optimizer over fully-batched session tensors), re-implemented
directly in raw JAX + optax so it follows the same ModelInterface contract as
RescorlaWagnerModel and SimpleRNNModel.

Input at trial t (matches the workshop's GRU):
    [a_{t-1} one-hot (max_n_arms),
     choices_available (max_n_arms),
     s_t one-hot (max_n_states),
     s_{t-1} one-hot (max_n_states),
     r_{t-1} / reward_scale (1)]

Key design points:
- Fixed output space of max_n_arms with an available-actions mask, so one model can
  be trained/evaluated across tasks with different n_actions (pass n_actions_list /
  n_actions explicitly when mixing task sizes).
- States are expected as integer label indices in 0..max_n_states-1 (Azulejos
  global label space is 0..27; DataManager stores these in the 'states_idx'
  column). Labels are kept global (not per-session normalized) because tasks with
  block_change_type == 'n_states' relabel states every block.
- The first trial of each session is excluded from the loss (zero prev-inputs),
  matching the workshop; probabilities are still returned for every trial.
- Reset/update gates are recorded during the forward pass and exposed via
  get_gate_history() for mechanism analysis.
- Optional minibatching over sessions (batch_size) for large multi-task runs;
  defaults to a single full batch to preserve the single-task baseline behaviour.
"""


class SimpleGRUModel(ModelInterface):
    def __init__(
        self,
        task_properties: dict,
        hidden_size: int = 16,
        learning_rate: float = 0.01,
        n_epochs: int = 50,
        seed: int = 0,
        max_n_arms: int = 6,
        max_n_states: int = 30,
        batch_size: int = None,
        reward_scale: float = 100.0,
        drop_first_trial: bool = True,
    ):
        super().__init__(task_properties)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        self.max_n_arms = int(max_n_arms)
        self.max_n_states = int(max_n_states)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.reward_scale = float(reward_scale)
        self.drop_first_trial = bool(drop_first_trial)

        # prev_choice + choices_available + s_t + s_{t-1} + prev_reward
        self.input_size = 2 * self.max_n_arms + 2 * self.max_n_states + 1
        self.output_size = self.max_n_arms

        self.params = self._initialize_parameters()
        self.loss_history = []
        self.test_loss_history = []
        self.hidden_history = None
        self.gate_history = None

    def _initialize_parameters(self):
        rng = np.random.default_rng(self.seed)
        scale = 0.1
        hidden, inp, out = self.hidden_size, self.input_size, self.output_size

        def w(shape):
            return jnp.array(rng.normal(scale=scale, size=shape))

        return {
            "Wxz": w((hidden, inp)), "Whz": w((hidden, hidden)), "bz": jnp.zeros((hidden,)),
            "Wxr": w((hidden, inp)), "Whr": w((hidden, hidden)), "br": jnp.zeros((hidden,)),
            "Wxh": w((hidden, inp)), "Whh": w((hidden, hidden)), "bh": jnp.zeros((hidden,)),
            "Wy": w((out, hidden)), "by": jnp.zeros((out,)),
        }

    def _build_batch(self, actions_list: list, rewards_list: list,
                     states_list: list = None, n_actions_list: list = None):
        """Pads variable-length sessions into one (session, time, feature) tensor
        plus a valid-trial mask and an available-actions mask."""
        max_len = max(len(np.asarray(a, dtype=int)) for a in actions_list)
        n_sessions = len(actions_list)

        X = np.zeros((n_sessions, max_len, self.input_size), dtype=float)
        Y = np.zeros((n_sessions, max_len), dtype=int)
        mask = np.zeros((n_sessions, max_len), dtype=bool)
        avail = np.zeros((n_sessions, max_len, self.output_size), dtype=bool)

        s_off = 2 * self.max_n_arms          # start of the s_t one-hot block
        ps_off = s_off + self.max_n_states   # start of the s_{t-1} one-hot block

        for i, (actions, rewards) in enumerate(zip(actions_list, rewards_list)):
            actions = np.asarray(actions, dtype=int)
            rewards = np.asarray(rewards, dtype=float)
            n_trials = len(actions)

            n_actions = self.n_actions if n_actions_list is None else int(n_actions_list[i])
            n_actions = min(n_actions, self.max_n_arms)

            # choices_available is constant per session in Azulejos (first n_actions arms)
            X[i, :, self.max_n_arms:self.max_n_arms + n_actions] = 1.0
            avail[i, :, :n_actions] = True

            if states_list is not None and states_list[i] is not None:
                states = np.asarray(states_list[i], dtype=int)
                states = np.clip(states, 0, self.max_n_states - 1)
            else:
                states = np.zeros(n_trials, dtype=int)

            # trial 0: current state is part of the input (feeds h_1 even if the
            # first trial is dropped from the loss)
            if n_trials > 0:
                s0 = int(states[0])
                if 0 <= s0 < self.max_n_states:
                    X[i, 0, s_off + s0] = 1.0

            for t in range(1, n_trials):
                prev_action = int(actions[t - 1])
                if 0 <= prev_action < self.max_n_arms:
                    X[i, t, prev_action] = 1.0
                s_t = int(states[t])
                if 0 <= s_t < self.max_n_states:
                    X[i, t, s_off + s_t] = 1.0
                s_prev = int(states[t - 1])
                if 0 <= s_prev < self.max_n_states:
                    X[i, t, ps_off + s_prev] = 1.0
                X[i, t, -1] = rewards[t - 1] / self.reward_scale

            Y[i, :n_trials] = np.clip(actions, 0, self.max_n_arms - 1)
            mask[i, :n_trials] = True

        if self.drop_first_trial:
            mask[:, 0] = False  # matches the workshop: no prediction from zero prev-inputs

        return jnp.array(X), jnp.array(Y), jnp.array(mask), jnp.array(avail)

    def _forward_logits(self, params, x_batch: jnp.ndarray):
        """Vectorized GRU forward pass: one lax.scan over time, batched across all
        sessions. Returns logits, hidden states, update gates (z) and reset gates (r)."""
        x_time_major = jnp.transpose(x_batch, (1, 0, 2))  # (time, batch, input)

        def step_fn(h_prev, x_t):
            z = jax.nn.sigmoid(x_t @ params["Wxz"].T + h_prev @ params["Whz"].T + params["bz"])
            r = jax.nn.sigmoid(x_t @ params["Wxr"].T + h_prev @ params["Whr"].T + params["br"])
            h_tilde = jnp.tanh(x_t @ params["Wxh"].T + (r * h_prev) @ params["Whh"].T + params["bh"])
            h_t = (1 - z) * h_prev + z * h_tilde
            logits_t = h_t @ params["Wy"].T + params["by"]
            return h_t, (logits_t, h_t, z, r)

        batch_size = x_batch.shape[0]
        h0 = jnp.zeros((batch_size, self.hidden_size))
        _, (logits_seq, hidden_seq, z_seq, r_seq) = lax.scan(step_fn, h0, x_time_major)
        logits = jnp.transpose(logits_seq, (1, 0, 2))  # (batch, time, n_actions)
        hidden = jnp.transpose(hidden_seq, (1, 0, 2))  # (batch, time, hidden)
        z_gates = jnp.transpose(z_seq, (1, 0, 2))
        r_gates = jnp.transpose(r_seq, (1, 0, 2))
        return logits, hidden, z_gates, r_gates

    def _masked_nll(self, params, X, Y, mask, avail):
        logits, _, _, _ = self._forward_logits(params, X)
        # mask out unavailable arms before softmax (fixed output space across tasks)
        masked_logits = jnp.where(avail, logits, -jnp.inf)
        log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
        one_hot = jax.nn.one_hot(Y, num_classes=self.output_size)
        trial_log_liks = jnp.sum(one_hot * log_probs, axis=-1)
        masked_log_liks = jnp.where(mask, trial_log_liks, 0.0)
        total_nll = -jnp.sum(masked_log_liks)
        n_valid = jnp.maximum(jnp.sum(mask), 1)
        return total_nll / n_valid

    def fit(
        self,
        actions_list: list,
        rewards_list: list,
        states_list: list = None,
        n_actions_list: list = None,
        val_actions_list: list = None,
        val_rewards_list: list = None,
        val_states_list: list = None,
        val_n_actions_list: list = None,
    ):
        """Fit GRU parameters with Adam over the training sessions.

        Optional state-aware inputs:
        - states_list: per-session state-index arrays (DataManager 'states_idx'),
          one int per trial, global labels in 0..max_n_states-1 (0..27 for
          Azulejos; do not normalize per session — see class docstring).
        - n_actions_list: per-session available-action counts; required when mixing
          tasks with different n_actions in one model.
        - val_*: held-out sessions whose NLL is tracked every epoch (mirrors the
          train/test loss curve in Sarah's workshop notebook).
        """
        if len(actions_list) == 0:
            raise ValueError("actions_list is empty; cannot fit model.")

        X, Y, mask, avail = self._build_batch(actions_list, rewards_list, states_list, n_actions_list)

        track_val = val_actions_list is not None and len(val_actions_list) > 0
        if track_val:
            X_val, Y_val, mask_val, avail_val = self._build_batch(
                val_actions_list, val_rewards_list, val_states_list, val_n_actions_list
            )

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self.params)

        def objective(params, X, Y, mask, avail):
            return self._masked_nll(params, X, Y, mask, avail)

        value_and_grad_fn = jax.value_and_grad(objective)
        eval_loss_fn = jax.jit(objective)

        @jax.jit
        def train_step(params, opt_state, X, Y, mask, avail):
            loss_value, grads = value_and_grad_fn(params, X, Y, mask, avail)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value

        self.loss_history = []
        self.test_loss_history = []
        params = self.params
        rng = np.random.default_rng(self.seed + 1)
        n_sessions = X.shape[0]
        use_minibatch = self.batch_size is not None and self.batch_size < n_sessions

        for _ in range(self.n_epochs):
            if track_val:
                self.test_loss_history.append(float(eval_loss_fn(params, X_val, Y_val, mask_val, avail_val)))

            if use_minibatch:
                epoch_losses = []
                order = rng.permutation(n_sessions)
                for start in range(0, n_sessions, self.batch_size):
                    idx = order[start:start + self.batch_size]
                    params, opt_state, loss_value = train_step(params, opt_state, X[idx], Y[idx], mask[idx], avail[idx])
                    epoch_losses.append(float(loss_value))
                epoch_loss = float(np.mean(epoch_losses))
            else:
                params, opt_state, loss_value = train_step(params, opt_state, X, Y, mask, avail)
                epoch_loss = float(loss_value)
            self.loss_history.append(epoch_loss)

        self.params = params
        self.is_fitted = True

        final_loss = self.loss_history[-1] if self.loss_history else float(objective(params, X, Y, mask, avail))
        print(f"GRU fit complete. Mean NLL/trial: {final_loss:.4f}")
        return self

    def get_action_probabilities(self, actions: np.ndarray, rewards: np.ndarray,
                                 states: np.ndarray = None, n_actions: int = None) -> np.ndarray:
        """Return P(A_t | history up to t-1, s_<=t) for all trials in a sequence.

        Output shape is (N_trials, max_n_arms); arms outside the n_actions available
        ones have probability 0 (masked softmax).
        """
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        n_actions = self.n_actions if n_actions is None else int(n_actions)
        if len(actions) == 0:
            self.hidden_history = np.zeros((0, self.hidden_size), dtype=float)
            self.gate_history = {
                "reset": np.zeros((0, self.hidden_size), dtype=float),
                "update": np.zeros((0, self.hidden_size), dtype=float),
            }
            return np.zeros((0, self.output_size), dtype=float)

        X, _, _, avail = self._build_batch(
            [actions], [rewards],
            [states] if states is not None else None,
            [n_actions],
        )
        logits, hidden, z_gates, r_gates = self._forward_logits(self.params, X)
        masked_logits = jnp.where(avail, logits, -jnp.inf)
        probs = jax.nn.softmax(masked_logits, axis=-1)

        self.hidden_history = np.asarray(hidden[0])
        self.gate_history = {
            "reset": np.asarray(r_gates[0]),
            "update": np.asarray(z_gates[0]),
        }
        return np.asarray(probs[0])

    def get_latent_states(self):
        return self.hidden_history

    def get_gate_history(self) -> dict:
        """Returns {'reset': (N_trials, hidden), 'update': (N_trials, hidden)}
        from the most recent get_action_probabilities() pass."""
        return self.gate_history

    def get_num_parameters(self) -> int:
        total = 0
        for value in self.params.values():
            total += int(np.prod(value.shape))
        return total

    def get_model_description(self) -> dict:
        base_desc = super().get_model_description()
        base_desc["description"] = (
            "GRU policy model trained with Adam over batched session tensors. "
            "Input at trial t: [a_{t-1}, choices_available, s_t, s_{t-1}, r_{t-1}]; "
            "fixed output over max_n_arms with an available-actions mask; "
            "exposes hidden states and reset/update gates for analysis."
        )
        base_desc["hidden_size"] = f"{self.hidden_size}"
        base_desc["learning_rate"] = f"{self.learning_rate}"
        base_desc["epochs"] = f"{self.n_epochs}"
        base_desc["input_size"] = f"{self.input_size}"
        base_desc["batch_size"] = "full" if self.batch_size is None else f"{self.batch_size}"
        return base_desc
