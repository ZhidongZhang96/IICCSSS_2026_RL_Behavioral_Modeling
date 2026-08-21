"""utils_ground_truth.py -- helpers for the ground-truth training direction.

This is a deliberately separate module from `utils.py`: it backs
`main_ground_truth.ipynb`, where the GRU is trained on *optimal* behavior
(ground-truth targets) instead of human choices. Nothing here modifies the
human-behavior pipeline in `utils.py` / `main.ipynb`; it only reuses their
read-only helpers (splits, replay metrics, curves, latents, PCA, gates).

Key idea:
- Target: the optimal action per trial (`best_actions`, the dataset's per-state
  argmax of mean arm outcomes).
- Input history: an *optimal agent's* experience -- the same states, but with
  optimal actions and the rewards those actions would have yielded
  (`arm_outcomes[t][best_actions[t]]`).

`optimal_rows()` returns rows that are drop-in compatible with the helpers in
`utils.py` (they are pd.Series with `actions`/`rewards` replaced by the optimal
agent's history), so `session_metrics`, `collect_latents`, `replay_probs` and
the curve/PCA/gate functions work unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def optimal_rows(rows_or_df) -> list:
    """Ground-truth trajectories for a session DataFrame or row list.

    Each returned row is a pd.Series with the same metadata as the input, but
    `actions`/`rewards` replaced by the optimal agent's history:
    - actions  = `best_actions` (per-state argmax of mean arm outcomes);
    - rewards  = `arm_outcomes[t][best_actions[t]]` (the outcome the optimal
      action would have received on that trial), with a fallback to the
      session-level mean reward per action when per-trial outcomes are absent.
    """
    rows = _as_rows(rows_or_df)
    out = []
    for row in rows:
        best = np.asarray(row["best_actions"], dtype=int)
        if "arm_outcomes" in row.index and row["arm_outcomes"] is not None:
            aos = [np.asarray(ao, dtype=float) for ao in row["arm_outcomes"]]
            rews = np.array([
                aos[t][a] if a < len(aos[t]) else 0.0
                for t, a in enumerate(best)
            ], dtype=float)
        else:  # fallback: session mean reward of the best action
            apr = np.asarray(row["action_reward_probs"], dtype=float)
            rews = np.array([apr[a] if a < len(apr) else 0.0 for a in best], dtype=float)
        states = (np.asarray(row["states_idx"], dtype=int)
                  if "states_idx" in row.index and row["states_idx"] is not None
                  else np.zeros(len(best), dtype=int))
        rec = {k: v for k, v in row.items()}
        rec["actions"] = best
        rec["rewards"] = rews
        rec["states"] = (np.asarray(row["states"], dtype=int)
                         if "states" in row.index and row["states"] is not None
                         else states.copy())
        rec["states_idx"] = states
        out.append(pd.Series(rec))
    return out


def train_gru_optimal(train_df, val_df, hidden_size: int = 32,
                      learning_rate: float = 1e-3, n_epochs: int = 8,
                      batch_size: int = 64, seed: int = 0,
                      max_sessions: int = None, max_n_arms: int = 6,
                      max_n_states: int = 30, reward_scale: float = 100.0):
    """Train a shared GRU on ground-truth trajectories (optimal actions/rewards).
    Thin wrapper over utils.train_gru; returns (model, used_optimal_rows)."""
    from utils import train_gru
    return train_gru(
        pd.DataFrame(optimal_rows(train_df)),
        pd.DataFrame(optimal_rows(val_df)),
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
        max_sessions=max_sessions,
        max_n_arms=max_n_arms,
        max_n_states=max_n_states,
        reward_scale=reward_scale,
    )


def best_action_ev(row) -> np.ndarray:
    """Per-trial EV of the best arm under the dataset's convention (mean outcome
    per state, argmax), used as a probe target for "does h_t track values?"."""
    aos = [np.asarray(ao, dtype=float) for ao in row["arm_outcomes"]]
    states = np.asarray(row["states"], dtype=int)
    best = np.asarray(row["best_actions"], dtype=int)
    out = np.zeros(len(best), dtype=float)
    for s in np.unique(states):
        mask = states == s
        ev = np.mean(np.stack([aos[t] for t in np.where(mask)[0]]), axis=0)
        out[mask] = ev[np.clip(best[mask], 0, len(ev) - 1)]
    return out


def session_split_mask(values, frac: float, seed: int):
    """Boolean mask splitting unique session ids into train (frac) / val (1-frac)
    so probe train/val trials never come from the same session."""
    uniq = np.unique(values)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    train_set = set(uniq[:int(len(uniq) * frac)])
    return np.array([v in train_set for v in values])


def train_linear_regression_probe(X_train, y_train, X_val, y_val,
                                  ridge: float = 1e-2) -> dict:
    """Closed-form ridge regression readout of a continuous target (e.g. the EV
    of the best arm) from the hidden state. Returns weights and train/val R²."""
    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=float)
    Xva = np.asarray(X_val, dtype=float)
    yva = np.asarray(y_val, dtype=float)
    A = Xtr.T @ Xtr + ridge * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)

    def r2(X, y):
        pred = X @ w
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {"w": w, "train_r2": r2(Xtr, ytr), "val_r2": r2(Xva, yva)}


def closed_loop_simulate(model, row, n_episodes: int = 5, seed: int = 0,
                         max_n_arms: int = 6, max_n_states: int = 30,
                         reward_scale: float = 100.0,
                         forced_history=None) -> dict:
    """Single-pass closed-loop simulation of a GRU on one session.

    The model samples its own actions trial by trial; the reward is
    `arm_outcomes[t][chosen]` and the chosen action/reward become the next
    input (its own trajectory). This is the online counterpart of the
    teacher-forced replay: an early mistake changes the feedback the model sees
    (distribution shift), which the P(best) recovery curve cannot capture.

    Implementation note: this replicates the GRU input layout and forward math
    of `Model_GRU.SimpleGRUModel` (see `_build_batch` / `_forward_logits`) in
    numpy, but runs in a single O(T) pass instead of calling
    `get_action_probabilities` once per trial (which would be O(T^2)). Keep the
    two in sync; an equivalence check (forced_history == replay) is run in the
    project tests.

    `forced_history=(actions, rewards)`: if given, the model does not sample --
    it follows the provided history (used for the equivalence test and for
    comparing with replay).

    Returns dict with arrays per episode: actions, rewards, p_best (probability
    assigned to the optimal action before sampling), best_chosen.
    """
    if not hasattr(model, "params"):
        raise TypeError("closed_loop_simulate currently supports GRU models "
                        "(models exposing .params and .hidden_size).")
    best = np.asarray(row["best_actions"], dtype=int)
    n = len(best)
    if n == 0:
        return {"actions": [], "rewards": [], "p_best": [], "best_chosen": []}
    states = (np.asarray(row["states_idx"], dtype=int)
              if "states_idx" in row.index and row["states_idx"] is not None
              else np.zeros(n, dtype=int))
    aos = [np.asarray(ao, dtype=float) for ao in row["arm_outcomes"]]
    n_actions = int(row["n_actions"])
    P = {k: np.asarray(v) for k, v in model.params.items()}
    H = model.hidden_size
    s_off = 2 * max_n_arms          # start of the s_t one-hot block
    ps_off = s_off + max_n_states   # start of the s_{t-1} one-hot block
    dim = 2 * max_n_arms + 2 * max_n_states + 1

    def build_x(t, prev_action, prev_reward):
        x = np.zeros(dim)
        x[max_n_arms:max_n_arms + n_actions] = 1.0  # choices_available
        st = int(states[t])
        if 0 <= st < max_n_states:
            x[s_off + st] = 1.0
        if t >= 1:
            if 0 <= prev_action < max_n_arms:
                x[prev_action] = 1.0
            sp = int(states[t - 1])
            if 0 <= sp < max_n_states:
                x[ps_off + sp] = 1.0
            x[-1] = prev_reward / reward_scale
        return x

    def sigmoid(u):
        return 1.0 / (1.0 + np.exp(-u))

    use_forced = forced_history is not None
    if use_forced:
        f_actions = np.asarray(forced_history[0], dtype=int)
        f_rewards = np.asarray(forced_history[1], dtype=float)

    out = {"actions": [], "rewards": [], "p_best": [], "best_chosen": []}
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed + ep)
        h = np.zeros(H)
        acts = np.zeros(n, dtype=int)
        rews = np.zeros(n, dtype=float)
        pb = np.zeros(n, dtype=float)
        bc = np.zeros(n, dtype=bool)
        prev_action, prev_reward = 0, 0.0
        for t in range(n):
            x = build_x(t, prev_action, prev_reward)
            z = sigmoid(x @ P["Wxz"].T + h @ P["Whz"].T + P["bz"])
            r = sigmoid(x @ P["Wxr"].T + h @ P["Whr"].T + P["br"])
            h_tilde = np.tanh(x @ P["Wxh"].T + (r * h) @ P["Whh"].T + P["bh"])
            h = (1 - z) * h + z * h_tilde
            logits = h @ P["Wy"].T + P["by"]
            logits_masked = logits.copy()
            logits_masked[n_actions:] = -np.inf
            e = np.exp(logits_masked - logits_masked.max())
            probs = e / e.sum()
            pb[t] = probs[best[t]]
            if use_forced:
                a = int(f_actions[t])
            else:
                a = int(rng.choice(n_actions, p=probs[:n_actions]))
            acts[t] = a
            bc[t] = (a == best[t])
            if use_forced:
                rw = float(f_rewards[t])
            else:
                rw = float(aos[t][a]) if a < len(aos[t]) else 0.0
            rews[t] = rw
            prev_action, prev_reward = a, rw
        out["actions"].append(acts)
        out["rewards"].append(rews)
        out["p_best"].append(pb)
        out["best_chosen"].append(bc)
    return out


def closed_loop_evaluate(model, rows_or_df, n_episodes: int = 5, seed: int = 0,
                         max_trials: int = None, **kwargs) -> pd.DataFrame:
    """Closed-loop simulation over sessions -> long DataFrame with
    session_id, episode, trial, action, reward, p_best, best_chosen,
    cum_reward, best_action and task metadata."""
    records = []
    for sid, row in enumerate(_as_rows(rows_or_df)):
        ep = closed_loop_simulate(model, row, n_episodes=n_episodes, seed=seed, **kwargs)
        best = np.asarray(row["best_actions"], dtype=int)
        n = len(best)
        for e in range(n_episodes):
            cum = np.cumsum(ep["rewards"][e])
            for t in range(n):
                if max_trials is not None and t >= max_trials:
                    break
                records.append({
                    "session_id": sid,
                    "episode": e,
                    "trial": t,
                    "action": int(ep["actions"][e][t]),
                    "reward": float(ep["rewards"][e][t]),
                    "p_best": float(ep["p_best"][e][t]),
                    "best_chosen": bool(ep["best_chosen"][e][t]),
                    "cum_reward": float(cum[t]),
                    "best_action": int(best[t]),
                    "task_id": row.get("task_id"),
                    "visibility": row.get("visibility"),
                    "n_actions": int(row["n_actions"]),
                })
    return pd.DataFrame(records)


def _as_rows(x):
    """DataFrame -> list of row Series; otherwise pass through a row list."""
    if isinstance(x, pd.DataFrame):
        return [row for _, row in x.iterrows()]
    return list(x)
