"""utils.py -- shared helpers for the Phase 2 cross-task pipeline.

The notebook `main.ipynb` is intentionally kept declarative: heavy lifting
(design-level splits, replay evaluation, learning curves, latent/gate
collection, PCA, linear probes) lives here so cells stay readable and the same
functions can be reused from scripts or other notebooks.

Environment note: this module requires the `iiccsss-rl` env
(jax + optax + pandas + scipy + matplotlib + seaborn).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax

from Model_RescorlaWagner import RescorlaWagnerModel
from Model_GRU import SimpleGRUModel


# ---------------------------------------------------------------------------
# Data / splits
# ---------------------------------------------------------------------------

def split_task_designs(dm, train_frac: float = 0.7, val_frac: float = 0.15,
                       seed: int = 0, task_col: str = "task_number"):
    """Split sessions by task *design* (task_number), so that no design appears
    in more than one split.  Returns (train_df, val_df, test_df, summary_df)."""
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.")
    designs = np.sort(dm.df[task_col].dropna().unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(designs)
    n_train = int(round(len(designs) * train_frac))
    n_val = max(1, int(round(len(designs) * val_frac)))
    # never leave the test split empty
    if n_train + n_val >= len(designs):
        n_train = max(1, len(designs) - n_val - 1)
    train_designs = designs[:n_train]
    val_designs = designs[n_train:n_train + n_val]
    test_designs = designs[n_train + n_val:]

    train_df = dm.df[dm.df[task_col].isin(train_designs)].reset_index(drop=True)
    val_df = dm.df[dm.df[task_col].isin(val_designs)].reset_index(drop=True)
    test_df = dm.df[dm.df[task_col].isin(test_designs)].reset_index(drop=True)

    summary = pd.DataFrame({
        "split": ["train", "val", "test"],
        "n_designs": [len(train_designs), len(val_designs), len(test_designs)],
        "n_sessions": [len(train_df), len(val_df), len(test_df)],
        "n_trials": [
            int(train_df["actions"].apply(len).sum()),
            int(val_df["actions"].apply(len).sum()),
            int(test_df["actions"].apply(len).sum()),
        ],
    })
    return train_df, val_df, test_df, summary


def _as_rows(x):
    """Normalize a session DataFrame or an iterable of row Series into a list."""
    if isinstance(x, pd.DataFrame):
        return [row for _, row in x.iterrows()]
    return list(x)


def session_lists(rows_or_df) -> dict:
    """Unpack a session-level DataFrame into per-session model inputs.

    Returns dict with 'actions', 'rewards', 'states' (states_idx), 'n_actions'
    lists (one entry per session) plus the original 'rows' (list of Series).
    """
    actions_list, rewards_list, states_list, n_actions_list, rows = [], [], [], [], []
    for row in _as_rows(rows_or_df):
        acts = np.asarray(row["actions"], dtype=np.int64)
        if acts.size == 0:
            continue
        rews = np.asarray(row["rewards"], dtype=float)
        if "states_idx" in row.index and row["states_idx"] is not None:
            states = np.asarray(row["states_idx"], dtype=int)
        else:
            states = np.zeros(len(acts), dtype=int)
        actions_list.append(acts)
        rewards_list.append(rews)
        states_list.append(states)
        n_actions_list.append(int(row["n_actions"]))
        rows.append(row)
    return {
        "actions": actions_list,
        "rewards": rewards_list,
        "states": states_list,
        "n_actions": n_actions_list,
        "rows": rows,
    }


def feature_balance_table(train_df, val_df, test_df, features):
    """Per-split session counts for each value of each feature (for split QA)."""
    frames = []
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for feat in features:
            counts = df[feat].value_counts(dropna=False).rename("n_sessions")
            out = counts.reset_index()
            out.columns = ["value", "n_sessions"]
            out.insert(0, "split", name)
            out.insert(1, "feature", feat)
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Replay evaluation
# ---------------------------------------------------------------------------

def evaluate_model_on_sessions(model, rows) -> dict:
    """Teacher-forced NLL + pseudo-R2 on a list of session rows.

    Probabilities come from get_action_probabilities(actions, rewards, states,
    n_actions) with the model weights frozen; the model only consumes the
    observed history (replay), it never fits on these sessions.
    """
    return summarize_sessions(session_metrics(model, rows))


def session_metrics(model, rows) -> pd.DataFrame:
    """One forward pass per session; returns a DataFrame with per-session NLL,
    trial count, random-chance NLL and the task metadata (for feature-level
    aggregation without re-running the model)."""
    eps = 1e-8
    meta_cols = ["task_id", "subject_id", "visibility", "n_actions", "n_states",
                 "points_type", "probs_type", "points_relationship",
                 "probs_relationship", "block_change_level", "block_change_type"]
    records = []
    for row in _as_rows(rows):
        acts = np.asarray(row["actions"], dtype=np.int64)
        if acts.size == 0:
            continue
        rews = np.asarray(row["rewards"], dtype=float)
        states = (np.asarray(row["states_idx"], dtype=int)
                  if "states_idx" in row.index and row["states_idx"] is not None
                  else None)
        n_actions = int(row["n_actions"])
        probs = model.get_action_probabilities(acts, rews, states=states, n_actions=n_actions)
        chosen = probs[np.arange(len(acts)), np.clip(acts, 0, probs.shape[1] - 1)]
        chosen = np.clip(chosen, eps, 1.0)
        rec = {col: row.get(col) for col in meta_cols}
        rec["nll"] = float(-np.sum(np.log(chosen)))
        rec["n_trials"] = len(acts)
        rec["random_nll"] = len(acts) * np.log(n_actions)
        records.append(rec)
    return pd.DataFrame(records)


def summarize_sessions(detail: pd.DataFrame) -> dict:
    """Aggregate a session_metrics() DataFrame into NLL / NLL_per_trial /
    Pseudo_R2 (relative to the per-session chance model)."""
    total_nll = float(detail["nll"].sum())
    total_trials = int(detail["n_trials"].sum())
    random_nll = float(detail["random_nll"].sum())
    return {
        "NLL": total_nll,
        "n_trials": total_trials,
        "NLL_per_trial": total_nll / max(total_trials, 1),
        "Pseudo_R2": 1.0 - total_nll / random_nll if random_nll > 0 else np.nan,
    }


def replay_probs(model, rows) -> list:
    """One replay forward per session; returns the list of probability matrices
    so curves can be computed without re-running the model."""
    probs_list = []
    for row in _as_rows(rows):
        acts = np.asarray(row["actions"], dtype=np.int64)
        if acts.size == 0:
            probs_list.append(np.zeros((0, 0), dtype=float))
            continue
        rews = np.asarray(row["rewards"], dtype=float)
        states = (np.asarray(row["states_idx"], dtype=int)
                  if "states_idx" in row.index and row["states_idx"] is not None
                  else None)
        probs_list.append(np.asarray(model.get_action_probabilities(
            acts, rews, states=states, n_actions=int(row["n_actions"]))))
    return probs_list


def best_action_curve_from_observed_actions(row, max_trials: int = 60) -> np.ndarray:
    """Human curve: 0/1 whether the participant chose the best action per trial."""
    actions = np.asarray(row["actions"], dtype=int)
    best = np.asarray(row["best_actions"], dtype=int)
    n = min(len(actions), len(best), max_trials)
    return (actions[:n] == best[:n]).astype(float)


def best_action_prob_curve_from_model(model, row, max_trials: int = 60) -> np.ndarray:
    """Model curve: P(best action) per trial under replay."""
    return best_action_prob_curve_from_probs(
        model.get_action_probabilities(
            np.asarray(row["actions"], dtype=int),
            np.asarray(row["rewards"], dtype=float),
            states=(np.asarray(row["states_idx"], dtype=int)
                    if "states_idx" in row.index and row["states_idx"] is not None
                    else None),
            n_actions=int(row["n_actions"])),
        row, max_trials)


def best_action_prob_curve_from_probs(probs: np.ndarray, row, max_trials: int = 60) -> np.ndarray:
    """Best-action probability curve from an already-computed probability matrix."""
    best = np.asarray(row["best_actions"], dtype=int)
    n = min(probs.shape[0], len(best), max_trials)
    return np.asarray(probs[np.arange(n), best[:n]], dtype=float)


def pull_average_learning_curve(curves, max_trials: int = None):
    """NaN-pad variable-length curves and return (mean, se) per trial."""
    if len(curves) == 0:
        return np.array([]), np.array([])
    max_len = max(len(c) for c in curves)
    if max_trials is not None:
        max_len = min(max_len, max_trials)
    padded = []
    for c in curves:
        c = np.asarray(c, dtype=float)[:max_len]
        padded.append(np.concatenate([c, np.full(max_len - len(c), np.nan)]))
    arr = np.asarray(padded, dtype=float)
    mean = np.nanmean(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
    return mean, se


# ---------------------------------------------------------------------------
# Model training wrappers
# ---------------------------------------------------------------------------

def _subsample_rows(rows, max_sessions: int, seed: int):
    rows = _as_rows(rows)
    if max_sessions is not None and len(rows) > max_sessions:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=max_sessions, replace=False)
        rows = [rows[i] for i in idx]
    return rows


def fit_rw_pooled(train_rows, seed: int = 0, max_sessions: int = None):
    """Fit one Rescorla-Wagner (alpha, beta) on a pool of sessions that may mix
    tasks with different n_actions. Returns (model, used_rows)."""
    rows = _subsample_rows(train_rows, max_sessions, seed)
    pack = session_lists(pd.DataFrame(rows))
    max_na = int(max(pack["n_actions"]))
    model = RescorlaWagnerModel({"n_actions": max_na})
    model.fit(pack["actions"], pack["rewards"], n_actions_list=pack["n_actions"])
    return model, rows


def train_gru(train_rows, val_rows, hidden_size: int = 32, learning_rate: float = 1e-3,
              n_epochs: int = 8, batch_size: int = 64, seed: int = 0,
              max_sessions: int = None, max_n_arms: int = 6,
              max_n_states: int = 30, reward_scale: float = 100.0):
    """Train the shared GRU on many task designs, tracking val NLL per epoch."""
    train_rows = _subsample_rows(train_rows, max_sessions, seed)
    tr = session_lists(pd.DataFrame(train_rows))
    va = session_lists(pd.DataFrame(val_rows))
    model = SimpleGRUModel(
        {"n_actions": max_n_arms},
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        seed=seed,
        max_n_arms=max_n_arms,
        max_n_states=max_n_states,
        batch_size=batch_size,
        reward_scale=reward_scale,
    )
    model.fit(
        tr["actions"], tr["rewards"], tr["states"], tr["n_actions"],
        val_actions_list=va["actions"], val_rewards_list=va["rewards"],
        val_states_list=va["states"], val_n_actions_list=va["n_actions"],
    )
    return model, train_rows


# ---------------------------------------------------------------------------
# Latent / gate collection
# ---------------------------------------------------------------------------

def collect_latents(model, rows) -> pd.DataFrame:
    """Run the frozen model over sessions (replay) and return one long-format
    row per trial with hidden states (h_k), update (z_k) and reset (r_k) gates,
    plus task metadata for condition-level analyses."""
    records = []
    for sid, row in enumerate(_as_rows(rows)):
        acts = np.asarray(row["actions"], dtype=np.int64)
        if acts.size == 0:
            continue
        rews = np.asarray(row["rewards"], dtype=float)
        states = (np.asarray(row["states_idx"], dtype=int)
                  if "states_idx" in row.index and row["states_idx"] is not None
                  else np.zeros(len(acts), dtype=int))
        model.get_action_probabilities(acts, rews, states=states,
                                       n_actions=int(row["n_actions"]))
        h = np.asarray(model.get_latent_states(), dtype=float)
        gates = model.get_gate_history()
        z = np.asarray(gates["update"], dtype=float)
        r = np.asarray(gates["reset"], dtype=float)
        best = np.asarray(row["best_actions"], dtype=int)
        stay = np.concatenate([[False], acts[1:] == acts[:-1]])
        n = len(acts)
        d = h.shape[1]
        for t in range(n):
            rec = {
                "session_id": sid,
                "subject_id": row.get("subject_id"),
                "task_id": row.get("task_id"),
                "trial": t,
                "state": int(states[t]),
                "action": int(acts[t]),
                "reward": float(rews[t]),
                "best_action": int(best[t]),
                "best_chosen": bool(acts[t] == best[t]),
                "stay": bool(stay[t]),
                "visibility": row.get("visibility"),
                "n_actions": int(row["n_actions"]),
                "n_states": row.get("n_states"),
                "points_type": row.get("points_type"),
                "probs_type": row.get("probs_type"),
                "points_relationship": row.get("points_relationship"),
                "probs_relationship": row.get("probs_relationship"),
                "block_change_level": row.get("block_change_level"),
                "block_change_type": row.get("block_change_type"),
            }
            for k in range(d):
                rec[f"h{k}"] = h[t, k]
                rec[f"z{k}"] = z[t, k]
                rec[f"r{k}"] = r[t, k]
            records.append(rec)
    return pd.DataFrame(records)


def hidden_columns(latent_df: pd.DataFrame) -> list:
    return [c for c in latent_df.columns if c.startswith("h")]


def gate_columns(latent_df: pd.DataFrame, gate: str) -> list:
    prefix = {"update": "z", "reset": "r"}.get(gate, gate)
    return [c for c in latent_df.columns if c.startswith(prefix)]


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca_scores(X, n_components: int = 2, center: bool = True, scale: bool = False):
    """PCA via SVD. Returns dict with scores, components, explained variance
    ratio, mean (and std when scaling). No sklearn dependency."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1])
    Xc = X - mean
    scale_vec = None
    if scale:
        scale_vec = Xc.std(axis=0)
        scale_vec[scale_vec == 0] = 1.0
        Xc = Xc / scale_vec
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, Xc.shape[1])
    explained = (S ** 2) / np.sum(S ** 2)
    return {
        "scores": U[:, :k] * S[:k],
        "components": Vt[:k],
        "explained_var_ratio": explained[:k],
        "mean": mean,
        "scale": scale_vec,
    }


def balanced_subset(latent_df: pd.DataFrame, by: str = "visibility",
                    n_per_group: int = 600, seed: int = 0) -> pd.DataFrame:
    """Balanced subsample: keep at most n_per_group trials per value of `by`
    (e.g. per visibility type) so embeddings are not dominated by the most
    frequent condition."""
    parts = []
    for i, (value, grp) in enumerate(latent_df.groupby(by)):
        if len(grp) <= n_per_group:
            parts.append(grp)
            continue
        rng = np.random.default_rng(seed + i)
        idx = rng.choice(len(grp), size=n_per_group, replace=False)
        parts.append(grp.iloc[np.sort(idx)])
    return pd.concat(parts, ignore_index=True)


def embed_2d(X, method: str = "umap", seed: int = 0, n_neighbors: int = 15,
             min_dist: float = 0.1, perplexity: int = 30, metric: str = "euclidean"):
    """2-D embedding of a (N, D) feature matrix with UMAP or t-SNE.

    method='umap' uses umap-learn; method='tsne' uses openTSNE. Both are used
    on a subsample of trials (see balanced_subset) to keep runtime manageable.
    """
    X = np.asarray(X, dtype=float)
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            n_components=2, random_state=seed, metric=metric)
        return np.asarray(reducer.fit_transform(X))
    if method == "tsne":
        from openTSNE import TSNE
        embedder = TSNE(perplexity=perplexity, random_state=seed,
                        n_jobs=1, verbose=False)
        return np.asarray(embedder.fit(X))
    raise ValueError(f"Unknown embedding method '{method}' (use 'umap' or 'tsne').")


# ---------------------------------------------------------------------------
# Linear probes (softmax readout of the hidden state)
# ---------------------------------------------------------------------------

def train_linear_probe(X_train, y_train, X_val, y_val, n_classes: int = None,
                       avail_train=None, avail_val=None, seed: int = 0,
                       epochs: int = 300, lr: float = 0.05,
                       batch_size: int = 256) -> dict:
    """Linear (no hidden layer) softmax probe with masked cross-entropy, fit by
    Adam on minibatches. Returns W, b, train/val accuracy, loss history."""
    Xtr = jnp.asarray(X_train, dtype=jnp.float32)
    ytr = jnp.asarray(y_train, dtype=jnp.int32)
    Xva = jnp.asarray(X_val, dtype=jnp.float32)
    yva = jnp.asarray(y_val, dtype=jnp.int32)
    n_classes = int(n_classes if n_classes is not None else y_train.max() + 1)
    n_feat = Xtr.shape[1]
    if avail_train is None:
        avail_train = np.ones((len(y_train), n_classes), dtype=bool)
    if avail_val is None:
        avail_val = np.ones((len(y_val), n_classes), dtype=bool)
    avtr = jnp.asarray(avail_train, dtype=bool)
    avva = jnp.asarray(avail_val, dtype=bool)

    rng = np.random.default_rng(seed)
    params = {
        "W": jnp.asarray(rng.normal(0, 0.05, size=(n_classes, n_feat)), dtype=jnp.float32),
        "b": jnp.zeros((n_classes,), dtype=jnp.float32),
    }
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_fn(p, X, y, avail):
        logits = X @ p["W"].T + p["b"]
        logits = jnp.where(avail, logits, -jnp.inf)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(jnp.take_along_axis(logp, y[:, None], axis=-1))

    @jax.jit
    def step(p, s, X, y, avail):
        loss, grads = jax.value_and_grad(loss_fn)(p, X, y, avail)
        updates, s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    n = Xtr.shape[0]
    losses = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            params, opt_state, loss = step(params, opt_state, Xtr[idx], ytr[idx], avtr[idx])
            epoch_loss.append(float(loss))
        losses.append(float(np.mean(epoch_loss)))

    def accuracy(p, X, y, avail):
        logits = X @ p["W"].T + p["b"]
        logits = jnp.where(avail, logits, -jnp.inf)
        pred = jnp.argmax(logits, axis=-1)
        return float(jnp.mean(pred == y))

    return {
        "W": np.asarray(params["W"]),
        "b": np.asarray(params["b"]),
        "train_acc": accuracy(params, Xtr, ytr, avtr),
        "val_acc": accuracy(params, Xva, yva, avva),
        "losses": losses,
        "n_classes": n_classes,
    }


def _session_split_mask(values, frac: float, seed: int):
    uniq = np.unique(values)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    train_set = set(uniq[:int(len(uniq) * frac)])
    return np.array([v in train_set for v in values])


def _subsample_sessions(latent_df: pd.DataFrame, max_sessions: int, seed: int):
    """Keep only a random subset of sessions (all their trials), so probe
    train/val splits never share trials from the same session."""
    sids = np.unique(latent_df["session_id"].values)
    if len(sids) <= max_sessions:
        return latent_df
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(sids, size=max_sessions, replace=False).tolist())
    return latent_df[latent_df["session_id"].isin(keep)].reset_index(drop=True)


def run_feature_probe(latent_df, feature: str, hidden_dim: int, seed: int = 0,
                      split_frac: float = 0.8, max_sessions: int = 80) -> dict:
    """Decode a task feature from h_t. Train/val split is by session, so no
    same-session trials leak across the probe's two sets."""
    latent_df = _subsample_sessions(latent_df, max_sessions, seed)
    X = latent_df[hidden_columns(latent_df)].values.astype(np.float32)
    y, classes = pd.factorize(latent_df[feature])
    mask = _session_split_mask(latent_df["session_id"].values, split_frac, seed)
    result = train_linear_probe(X[mask], y[mask], X[~mask], y[~mask],
                                n_classes=len(classes), seed=seed)
    result["feature"] = feature
    result["classes"] = classes.tolist()
    result["n_train_trials"] = int(mask.sum())
    result["n_val_trials"] = int((~mask).sum())
    return result


def run_action_probe(latent_df, hidden_dim: int, seed: int = 0,
                     split_frac: float = 0.8, max_sessions: int = 80) -> dict:
    """Decode the action taken at trial t from h_t, masking unavailable arms."""
    latent_df = _subsample_sessions(latent_df, max_sessions, seed)
    X = latent_df[hidden_columns(latent_df)].values.astype(np.float32)
    y = latent_df["action"].values.astype(np.int32)
    avail = np.zeros((len(y), 6), dtype=bool)
    for na, idx in latent_df.groupby("n_actions").groups.items():
        avail[idx, :int(na)] = True
    mask = _session_split_mask(latent_df["session_id"].values, split_frac, seed)
    result = train_linear_probe(X[mask], y[mask], X[~mask], y[~mask],
                                n_classes=6, avail_train=avail[mask],
                                avail_val=avail[~mask], seed=seed)
    result["feature"] = "next_action"
    result["n_train_trials"] = int(mask.sum())
    result["n_val_trials"] = int((~mask).sum())
    return result


def run_binary_probe(latent_df, target: str, hidden_dim: int, seed: int = 0,
                     split_frac: float = 0.8, max_sessions: int = 80) -> dict:
    """Decode a binary behavioral target (e.g. 'stay' or 'best_chosen') from h_t."""
    latent_df = _subsample_sessions(latent_df, max_sessions, seed)
    X = latent_df[hidden_columns(latent_df)].values.astype(np.float32)
    y = latent_df[target].astype(np.int32).values
    mask = _session_split_mask(latent_df["session_id"].values, split_frac, seed)
    result = train_linear_probe(X[mask], y[mask], X[~mask], y[~mask],
                                n_classes=2, seed=seed)
    result["feature"] = target
    result["n_train_trials"] = int(mask.sum())
    result["n_val_trials"] = int((~mask).sum())
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_learning_curves(curves: dict, title: str = "", ylabel: str = "",
                         xlabel: str = "Trial", max_trials: int = 60,
                         ax=None, legend_kwargs: dict = None):
    """curves: {label: (mean, se)}. Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(max_trials)
    for label, (mean, se) in curves.items():
        n = min(len(mean), max_trials)
        ax.errorbar(x[:n], mean[:n], yerr=se[:n], fmt="-o", capsize=3,
                    label=label, markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or "metric")
    ax.set_title(title)
    ax.legend(**(legend_kwargs or {}))
    return ax


def plot_model_curves_by_feature(curves_by_model: dict, feature: str,
                                 max_trials: int = 60, ax=None,
                                 cmaps: dict = None):
    """Per-feature learning curves with one colour family per model.

    curves_by_model: {model: {feature_value: (mean, se)}} — for each model, one
    curve per value of the feature. Values are drawn as different shades of the
    model's colormap, so model identity (Human/RW/GRU) reads from the colour
    family while the feature value reads from the shade.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    cmaps = cmaps or {"Human": "Blues", "Rescorla-Wagner": "Oranges", "GRU": "Purples"}
    x = np.arange(max_trials)
    for model, by_value in curves_by_model.items():
        cmap = plt.get_cmap(cmaps.get(model, "Greys"))
        values = sorted(by_value.keys(), key=str)
        if len(values) == 0:
            continue
        shades = np.linspace(0.35, 0.95, len(values)) if len(values) > 1 else [0.7]
        for shade, value in zip(shades, values):
            mean, se = by_value[value]
            n = min(len(mean), max_trials)
            color = cmap(float(shade))
            label = f"{model} · {value}" if len(values) > 1 else str(model)
            ax.errorbar(x[:n], mean[:n], yerr=se[:n], fmt="-o", capsize=2,
                        color=color, label=label, markersize=2.5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("best-action metric")
    ax.set_title(f"by {feature}")
    ax.legend(fontsize=7, ncol=2)
    return ax


def plot_scatter_by_feature(df: pd.DataFrame, x_col: str, y_col: str,
                            feature: str, ax=None, title: str = None):
    """Scatter of an embedding / 2-D projection colored by a categorical feature."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    palette = dict(zip(
        sorted(df[feature].dropna().unique(), key=str),
        plt.cm.tab10.colors,
    ))
    for value, grp in df.groupby(feature, dropna=False):
        color = palette.get(value, "gray")
        label = "NaN" if (isinstance(value, float) and np.isnan(value)) else str(value)
        ax.scatter(grp[x_col], grp[y_col], s=6, alpha=0.45, color=color, label=label)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"by {feature}")
    ax.legend(markerscale=3, fontsize=8)
    return ax


def plot_pca_by_feature(scores_df: pd.DataFrame, feature: str,
                        pcx: str = "PC1", pcy: str = "PC2", ax=None):
    """Scatter of PCA scores colored by a task feature."""
    return plot_scatter_by_feature(
        scores_df, pcx, pcy, feature, ax=ax,
        title=f"Hidden-state PCA by {feature}")


def plot_gate_curves(latent_df: pd.DataFrame, gate: str = "update",
                     feature: str = None, max_trials: int = 60,
                     title: str = None, ax=None):
    """Mean gate value over units, per trial, optionally split by a feature.
    gate is 'update' (z) or 'reset' (r)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    cols = gate_columns(latent_df, gate)
    gate_df = latent_df[["session_id", "trial", feature] + cols].copy() if feature else \
        latent_df[["session_id", "trial"] + cols].copy()
    gate_df["gate"] = gate_df[cols].mean(axis=1)
    gate_df = gate_df[gate_df["trial"] < max_trials]
    if feature:
        groups = gate_df.groupby(feature)
    else:
        groups = [("all", gate_df)]
    for label, grp in groups:
        mean, se = pull_average_learning_curve(
            [g["gate"].values for _, g in grp.groupby("session_id")], max_trials=max_trials)
        ax.errorbar(np.arange(len(mean)), mean, yerr=se, fmt="-o", capsize=3,
                    label=str(label), markersize=3)
    gate_name = "update gate z" if gate == "update" else "reset gate r"
    ax.set_xlabel("Trial")
    ax.set_ylabel(f"mean {gate_name}")
    ax.set_title(title or f"Gate dynamics ({gate_name}) by {feature or 'all sessions'}")
    ax.legend(fontsize=8)
    return ax


def plot_probe_results(results: list, ax=None):
    """Bar chart of probe validation accuracies vs chance."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    rows = [{"probe": r["feature"], "val_acc": r["val_acc"], "train_acc": r["train_acc"]}
            for r in results]
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["train_acc"], width=0.4, label="train acc", color="#4C72B0")
    ax.bar(x + 0.2, df["val_acc"], width=0.4, label="val acc", color="#DD8452")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["probe"], rotation=20, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title("Linear probes on hidden states (val split by session)")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Mechanism analysis helpers (main_mechanism.ipynb: Directions 1 & 2)
# ---------------------------------------------------------------------------

def lda_projection(X, y, n_components: int = 2, ridge: float = 1e-3):
    """Multiclass LDA projection in numpy (no sklearn).

    Returns dict with 'scores' (N, k), 'directions' (k, D), 'eigenvalues',
    'between_within_ratio' (trace S_b / trace S_w) and 'classes'.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    mu = X.mean(axis=0)
    S_b = np.zeros((X.shape[1], X.shape[1]))
    S_w = np.zeros_like(S_b)
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        d = (mc - mu)[:, None]
        S_b += len(Xc) * (d @ d.T)
        Xcc = Xc - mc
        S_w += Xcc.T @ Xcc
    M = np.linalg.pinv(S_w + ridge * np.eye(X.shape[1])) @ S_b
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    k = min(n_components, X.shape[1])
    directions = vecs[:, :k].T  # (k, D)
    return {
        "scores": X @ directions.T,
        "directions": directions,
        "eigenvalues": vals[:k],
        "between_within_ratio": float(np.trace(S_b) / (np.trace(S_w) + 1e-12)),
        "classes": classes.tolist(),
    }


def unit_selectivity(X, y, eps: float = 1e-8) -> np.ndarray:
    """Per-unit cross-condition selectivity in [0, 1]:
    (max_c mean_c - min_c mean_c) / (max_c mean_c + min_c mean_c + eps)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    means = np.stack([X[y == c].mean(axis=0) for c in np.unique(y)], axis=0)
    mx = means.max(axis=0)
    mn = means.min(axis=0)
    return (mx - mn) / (mx + mn + eps)


def top_k_decodability(latent_df: pd.DataFrame, feature: str, k: int,
                       seed: int = 0, max_sessions: int = 80) -> dict:
    """Val accuracy of a linear probe decoding `feature` from the top-k most
    condition-selective hidden units (vs the full hidden state)."""
    sub = _subsample_sessions(latent_df, max_sessions, seed)
    X = sub[hidden_columns(sub)].values.astype(np.float32)
    y, classes = pd.factorize(sub[feature])
    mask = _session_split_mask(sub["session_id"].values, 0.8, seed)
    sel = unit_selectivity(X[mask], y[mask])
    top = np.argsort(sel)[::-1][:k]
    res = train_linear_probe(X[mask][:, top], y[mask], X[~mask][:, top], y[~mask],
                             n_classes=len(classes), seed=seed)
    return {
        "k": int(k),
        "train_acc": res["train_acc"],
        "val_acc": res["val_acc"],
        "chance": 1.0 / len(classes),
        "top_units": top.tolist(),
        "selectivity_top": sel[top].tolist(),
    }


def per_session_gate_curves(latent_df: pd.DataFrame, gate: str = "update",
                            max_trials: int = 60, by: str = None) -> pd.DataFrame:
    """Long DataFrame (session_id, trial, value) of the gate mean over units,
    one curve per session — the raw material for gate-curve distances.
    If `by` is given, a unique (by, session_id) key is added so sessions from
    different groups can share session_id values (e.g. after concatenating
    per-condition latents)."""
    cols = gate_columns(latent_df, gate)
    keep = latent_df["trial"] < max_trials
    out = latent_df.loc[keep, ["session_id", "trial"] + cols].copy()
    out["value"] = out[cols].mean(axis=1)
    out = out[["session_id", "trial", "value"]]
    if by is not None:
        out[by] = latent_df.loc[keep, by].values
        out["_skey"] = (out[by].astype(str) + "_" +
                        out["session_id"].astype(str)).values
    return out.reset_index(drop=True)


def _pair_curve_distance(g1: pd.DataFrame, g2: pd.DataFrame) -> float:
    m1 = g1.groupby("trial")["value"].mean()
    m2 = g2.groupby("trial")["value"].mean()
    t = m1.index.intersection(m2.index)
    return float(np.mean(np.abs(m1.loc[t].values - m2.loc[t].values)))


def gate_curve_distances_within(latent_df: pd.DataFrame, gate: str = "update",
                                by: str = "visibility", max_trials: int = 60,
                                n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Pairwise between-condition gate-curve distances within one latent df,
    with bootstrap 95% CIs (resampling sessions per condition).
    Distance(c1, c2) = mean_t |mean_s z_c1(t) - mean_s z_c2(t)|."""
    curves = per_session_gate_curves(latent_df, gate, max_trials, by=by)
    groups = {c: g for c, g in curves.groupby(by)}
    rng = np.random.default_rng(seed)
    conds = sorted(groups.keys(), key=str)
    rows = []
    for i, c1 in enumerate(conds):
        for c2 in conds[i + 1:]:
            g1, g2 = groups[c1], groups[c2]
            s1 = g1["_skey"].unique()
            s2 = g2["_skey"].unique()
            boots = []
            for _ in range(n_boot):
                b1 = g1[g1["_skey"].isin(rng.choice(s1, len(s1), replace=True))]
                b2 = g2[g2["_skey"].isin(rng.choice(s2, len(s2), replace=True))]
                boots.append(_pair_curve_distance(b1, b2))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({
                "cond_a": str(c1), "cond_b": str(c2),
                "distance": _pair_curve_distance(g1, g2),
                "ci_low": float(lo), "ci_high": float(hi),
                "n_a": len(s1), "n_b": len(s2),
            })
    return pd.DataFrame(rows)


def gate_curve_distance_between(latent_a: pd.DataFrame, latent_b: pd.DataFrame,
                                gate: str = "update", max_trials: int = 60,
                                n_boot: int = 2000, seed: int = 0) -> dict:
    """Gate-curve distance between two models' latents on the SAME sessions
    (aligned by session_id), with bootstrap 95% CI over sessions."""
    ca = per_session_gate_curves(latent_a, gate, max_trials)
    cb = per_session_gate_curves(latent_b, gate, max_trials)
    sids = np.intersect1d(ca["session_id"].values, cb["session_id"].values)
    ca = ca[ca["session_id"].isin(sids)]
    cb = cb[cb["session_id"].isin(sids)]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = rng.choice(sids, len(sids), replace=True)
        boots.append(_pair_curve_distance(ca[ca["session_id"].isin(s)],
                                          cb[cb["session_id"].isin(s)]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "distance": _pair_curve_distance(ca, cb),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_sessions": int(len(sids)),
    }


def paired_nll_delta(model_a, model_b, rows) -> np.ndarray:
    """Per-session Delta NLL = model_a - model_b on the same sessions/order."""
    a = session_metrics(model_a, rows)["nll"].values
    b = session_metrics(model_b, rows)["nll"].values
    return np.asarray(a - b, dtype=float)


def bootstrap_ci(samples, stat=np.mean, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap 95% CI of a statistic over resampled observations."""
    samples = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    boots = [stat(samples[rng.integers(0, len(samples), len(samples))])
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(stat(samples)), "ci_low": float(lo), "ci_high": float(hi)}


def cca_alignment(H1, H2, k: int = None, ridge: float = 1e-3) -> dict:
    """Canonical correlations between two aligned hidden-state matrices (numpy)."""
    H1 = np.asarray(H1, dtype=float)
    H2 = np.asarray(H2, dtype=float)
    H1c = H1 - H1.mean(axis=0)
    H2c = H2 - H2.mean(axis=0)
    n = len(H1c)

    def _inv_sqrt(S):
        w, V = np.linalg.eigh(S)
        return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T

    S11 = H1c.T @ H1c / n + ridge * np.eye(H1c.shape[1])
    S22 = H2c.T @ H2c / n + ridge * np.eye(H2c.shape[1])
    S12 = H1c.T @ H2c / n
    M = _inv_sqrt(S11) @ S12 @ _inv_sqrt(S22)
    _, sv, _ = np.linalg.svd(M)
    k = k or min(H1c.shape[1], H2c.shape[1])
    corrs = np.clip(sv[:k], 0.0, 1.0)
    return {"canonical_corrs": corrs, "mean_top_k": float(corrs.mean())}


def probe_transfer(X_train, y_train, avail_train, val_dict, y_val, avail_val,
                   seed: int = 0) -> dict:
    """Train a masked-softmax linear probe on the shared model's h and apply the
    SAME readout to each model's h (val_dict maps name -> X_val)."""
    res = train_linear_probe(X_train, y_train, X_train, y_train,
                             n_classes=avail_train.shape[1],
                             avail_train=avail_train, avail_val=avail_train,
                             seed=seed)
    W, b = res["W"], res["b"]

    def _acc(X, y, avail):
        logits = np.asarray(X, dtype=float) @ W.T + b
        logits = np.where(avail, logits, -np.inf)
        pred = np.argmax(logits, axis=-1)
        return float(np.mean(pred == y))

    out = {"train_acc": _acc(X_train, y_train, avail_train)}
    for name, Xv in val_dict.items():
        out[name] = _acc(Xv, y_val, avail_val)
    return out


def train_gru_per_condition(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            by: str = "visibility", cap: int = None,
                            seed: int = 0, **gru_kwargs) -> dict:
    """Train one GRU per value of `by` (extensible to n_actions / points_type /
    probs_type / block_change_type by changing `by`). Sessions per condition are
    optionally capped (balanced). Returns {value: (model, used_train_rows)}."""
    out = {}
    for value, grp in train_df.groupby(by):
        grp = grp.reset_index(drop=True)
        if cap is not None and len(grp) > cap:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(grp), size=cap, replace=False))
            grp = grp.iloc[idx].reset_index(drop=True)
        va = val_df[val_df[by] == value].reset_index(drop=True)
        model, used = train_gru(grp, va, seed=seed, **gru_kwargs)
        out[value] = (model, used)
    return out


def _embed_panel_df(latent_all: pd.DataFrame, by: str, method: str,
                    n_per_group: int, seed: int) -> pd.DataFrame:
    """Balanced subsample + one embedding fit, returning a DataFrame with the
    2-D coordinates (PC1/PC2 for 'pca', METHOD_1/METHOD_2 otherwise)."""
    sub = balanced_subset(latent_all, by=by, n_per_group=n_per_group, seed=seed)
    X = sub[hidden_columns(sub)].values
    if method == "pca":
        coords = pca_scores(X, n_components=2)["scores"]
        c1, c2 = "PC1", "PC2"
    else:
        coords = embed_2d(X, method=method, seed=seed)
        c1, c2 = f"{method.upper()}_1", f"{method.upper()}_2"
    sub = sub.copy()
    sub[c1], sub[c2] = coords[:, 0], coords[:, 1]
    return sub


def per_condition_report(train_df: pd.DataFrame, val_df: pd.DataFrame,
                         test_df: pd.DataFrame, by: str, shared_model,
                         hidden_size: int = 32, learning_rate: float = 1e-3,
                         n_epochs: int = 8, batch_size: int = 64, seed: int = 0,
                         cap: int = None, min_test_sessions: int = 30,
                         max_trials: int = 60, n_boot: int = 2000,
                         embed_methods=("umap", "tsne"),
                         n_embed_per_group: int = 400) -> dict:
    """End-to-end Direction-2 report for one condition feature `by`.

    Trains one GRU per condition value (same hyperparameters as the shared
    model), collects latents for the shared and per-condition models on the
    same held-out sessions, and returns everything Direction 2 needs:
    per_models, cond_test, lat_shared/lat_per, paired_nll, gate_between,
    gate_among, probe_transfer, cca, embeddings (shared vs per-condition
    merged 2-D projections per method), lat_per_all (for gate curves).
    """
    per_models = train_gru_per_condition(
        train_df, val_df, by=by, cap=cap, seed=seed,
        hidden_size=hidden_size, learning_rate=learning_rate,
        n_epochs=n_epochs, batch_size=batch_size)
    cond_test = {v: [row for _, row in test_df[test_df[by] == v].iterrows()]
                 for v in per_models}

    lat_shared, lat_per = {}, {}
    for v, (model, _) in per_models.items():
        rows = cond_test[v]
        if len(rows) < min_test_sessions:
            continue
        lat_shared[v] = collect_latents(shared_model, rows)
        lat_per[v] = collect_latents(model, rows)

    # (a) paired NLL on the same held-out sessions
    paired = []
    for v in lat_per:
        rows = cond_test[v]
        shared_nll = session_metrics(shared_model, rows)["nll"].values
        per_nll = session_metrics(per_models[v][0], rows)["nll"].values
        ci = bootstrap_ci(shared_nll - per_nll, n_boot=n_boot, seed=seed)
        paired.append({"condition": v, "n_test": len(rows),
                       "delta_nll_mean": ci["mean"], "ci_low": ci["ci_low"],
                       "ci_high": ci["ci_high"]})
    paired_df = pd.DataFrame(paired)

    # (b) gate distances (per-condition vs shared, and among per-condition)
    lat_per_all = pd.concat(list(lat_per.values()), ignore_index=True)
    gate_between, gate_among = [], {}
    for gate, _label in [("update", "z"), ("reset", "r")]:
        for v in lat_per:
            d = gate_curve_distance_between(lat_shared[v], lat_per[v], gate=gate,
                                            max_trials=max_trials, n_boot=n_boot,
                                            seed=seed)
            gate_between.append({"condition": v, "gate": gate,
                                 "distance": d["distance"], "ci_low": d["ci_low"],
                                 "ci_high": d["ci_high"]})
        gate_among[gate] = gate_curve_distances_within(
            lat_per_all, gate=gate, by=by, max_trials=max_trials,
            n_boot=n_boot, seed=seed)
    gate_between_df = pd.DataFrame(gate_between)

    # (c) representation alignment (probe transfer + CCA)
    transfer, cca = [], []
    for v in lat_per:
        lat_s, lat_p = lat_shared[v], lat_per[v]
        X_s = lat_s[hidden_columns(lat_s)].values
        X_p = lat_p[hidden_columns(lat_p)].values
        y = lat_s["action"].values
        avail = np.zeros((len(lat_s), 6), dtype=bool)
        for na, idx in lat_s.groupby("n_actions").groups.items():
            avail[idx, :int(na)] = True
        mask = _session_split_mask(lat_s["session_id"].values, 0.8, seed)
        tr = probe_transfer(X_s[mask], y[mask], avail[mask],
                            {"per": X_p[mask]}, y[mask], avail[mask], seed=seed)
        transfer.append({"condition": v, "shared_train_acc": tr["train_acc"],
                         "per_transfer_acc": tr["per"]})
        c = cca_alignment(X_s, X_p, k=min(8, X_s.shape[1]))
        cca.append({"condition": v, "cca_top1": float(c["canonical_corrs"][0]),
                    "cca_mean_top8": c["mean_top_k"]})
    transfer_df = pd.DataFrame(transfer)
    cca_df = pd.DataFrame(cca)

    # (d) merged embeddings: shared model vs per-condition models
    lat_shared_all = pd.concat(list(lat_shared.values()), ignore_index=True)
    embeddings = {
        method: {
            "shared": _embed_panel_df(lat_shared_all, by, method, n_embed_per_group, seed),
            "per": _embed_panel_df(lat_per_all, by, method, n_embed_per_group, seed),
        }
        for method in embed_methods
    }

    return {
        "per_models": per_models,
        "cond_test": cond_test,
        "lat_shared": lat_shared,
        "lat_per": lat_per,
        "lat_per_all": lat_per_all,
        "paired_nll": paired_df,
        "gate_between": gate_between_df,
        "gate_among": gate_among,
        "probe_transfer": transfer_df,
        "cca": cca_df,
        "embeddings": embeddings,
        "by": by,
    }
