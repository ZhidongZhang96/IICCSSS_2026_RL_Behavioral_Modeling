import os
import json
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import osfclient
import jax.numpy as jnp

"""
File: DataManager.py
This module provides the DataManager class responsible for loading,
parsing, and managing the Azulejos behavioral dataset.
"""

class DataManager:

    @staticmethod
    def _clean_valid_trials(actions, rewards, states=None):
        """Drop sentinel invalid trials (e.g., first choice = -1) from each session."""
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=float)
        valid = actions >= 0
        if not np.all(valid):
            actions = actions[valid]
            rewards = rewards[valid]
        if states is not None:
            states = np.asarray(states, dtype=int)
            if not np.all(valid):
                states = states[valid]
            return actions, rewards, states
        return actions, rewards, None

    def __init__(self, data_dir: str = './azulejos_data', download_limit: int = 419):
        """
        Initializes the DataManager, fetches data from OSF if missing, and loads
        the dataset into a unified, session-based memory structure.
        Downloads the maximum of 419 files by default, but can be subset.

        NOTE: for the workshop we assume the dataset is already on the participant's
        device. Therefore, missing local data should not silently trigger a download
        unless the caller explicitly enables it by setting download_limit > 0.
        """

        self.data_dir = os.path.abspath(data_dir)
        self.download_limit = max(0, int(download_limit))
        self.max_n_arms = 6
        self.max_n_states = 5

        if not os.path.exists(self.data_dir) or len([f for f in os.listdir(self.data_dir) if f.endswith('.csv')]) == 0:
            if self.download_limit <= 0:
                raise FileNotFoundError(
                    "No local Azulejos CSV files were found. "
                    f"Expected files under: {self.data_dir}. "
                    "Please place the dataset on your device and update the path."
                )
            print("Local data not found. Initiating OSF download...")
            self._download_osf_data(limit=self.download_limit)
        else:
            n_files = len([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
            print(f"Local data found with {n_files} files. Skipping OSF download.")

        # Parse the raw CSVs into a nested pandas DataFrame
        self.df = self._load_and_parse_data()


    def _download_osf_data(self, limit: int):
        """
        Connects to the Open Science Framework (OSF) API to download raw behavioral data.
        """
        if limit <= 0:
            print("Download limit is 0; skipping OSF download. Please provide a local dataset.")
            return

        os.makedirs(self.data_dir, exist_ok=True)
        osf = osfclient.OSF()
        project = osf.project('g2ds8')  # Azulejos OSF Project ID

        downloaded_count = 0
        for storage in project.storages:
            all_files = [f for f in storage.files if f.name.endswith('.csv')]
            print(f"Found {len(all_files)} files on OSF. Downloading {limit} files...")

            for file_ in all_files:
                file_path = os.path.join(self.data_dir, file_.name)
                if not os.path.exists(file_path):
                    try:
                        with open(file_path, 'wb') as local_file:
                            file_.write_to(local_file)
                        downloaded_count += 1
                    except Exception as e:
                        print(f"Failed to download {file_.name} (Error: {e}). Skipping...")
                        # Clean up the empty/corrupted file created by 'wb'
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        continue

                if downloaded_count >= limit:
                    return

    def _load_and_parse_data(self) -> pd.DataFrame:
        """
        Parses raw jsPsych CSVs into a structured DataFrame where each row is a complete task session.
        Calculates ground-truth values and builds the 3D matrices required for RNNs.
        """
        all_data = []
        file_list = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.csv')]

        print('Formatting data for models...')
        for csv_file in tqdm(file_list):
            try:
                df_raw = pd.read_csv(csv_file)
            except Exception:
                continue

            # Ensure trials are sequentially ordered
            if 'trial_index' in df_raw.columns:
                df_raw = df_raw.sort_values(by='trial_index').reset_index(drop=True)

            if 'trial_epoch' not in df_raw.columns:
                continue

            n_tasks = len(df_raw[df_raw['trial_epoch'] == 'learn_choice']['session'].dropna().unique())

            # Loop over tasks to pull sequence arrays
            for task in range(1, n_tasks):
                task_df = df_raw[df_raw['session'] == task]
                choice_index = task_df['trial_epoch'] == 'learn_choice'
                n_trials = np.count_nonzero(choice_index)

                if n_trials == 0:
                    continue

                n_actions = int(task_df['n_actions'].dropna().unique()[0])
                choices = np.array(task_df['arm_chosen'][choice_index].values, dtype=np.int64)
                rewards = np.array(task_df['points_won'][choice_index].values, dtype=float)
                states = np.array(task_df['state'][choice_index].values, dtype=int) if 'state' in task_df.columns else np.zeros(n_trials, dtype=int)

                choices, rewards, states = self._clean_valid_trials(choices, rewards, states)
                if len(choices) == 0:
                    continue

                arm_outcomes = [json.loads(x) for x in task_df['arm_outcomes'][choice_index]]
                valid_choice_mask = np.array(task_df['arm_chosen'][choice_index].values, dtype=np.int64) >= 0
                arm_outcomes = [out for out, valid in zip(arm_outcomes, valid_choice_mask) if valid]
                if len(arm_outcomes) == 0:
                    continue

                EV_arms = np.zeros((len(np.unique(states)), np.shape(arm_outcomes)[1]))
                best_arm = np.zeros(len(choices), dtype=int)

                for i, state in enumerate(np.unique(states)):
                    state_mask = states == state
                    state_idx = np.where(state_mask)[0]
                    EV_arms[i] = np.mean([arm_outcomes[idx] for idx in state_idx], axis=0)
                    best_arm[state_mask] = np.argmax(EV_arms[i])

                # --- BUILD RNN TENSORS ---
                X_choices = np.zeros((len(choices), self.max_n_arms))
                X_choices[np.arange(len(choices)), choices] = 1
                X_prev_choices = np.roll(X_choices, 1, axis=0)
                X_prev_choices[0, :] = 0

                X_choices_avail = np.zeros((len(choices), self.max_n_arms))
                X_choices_avail[:, :n_actions] = 1

                X_states = np.zeros((len(choices), self.max_n_states))
                valid_states = np.clip(states, 0, self.max_n_states - 1)
                X_states[np.arange(len(choices)), valid_states] = 1
                X_prev_states = np.roll(X_states, 1, axis=0)
                X_prev_states[0, :] = 0

                prev_rewards = np.roll(rewards, 1).reshape(-1, 1)
                prev_rewards[0] = 0

                X = np.concatenate((X_prev_choices, X_choices_avail, X_states, X_prev_states, prev_rewards), axis=1)

                mean_reward_by_action = np.zeros(n_actions)
                for action_idx in range(n_actions):
                    action_mask = choices == action_idx
                    if np.any(action_mask):
                        mean_reward_by_action[action_idx] = np.mean(rewards[action_mask])

                # TODO: 这里只包含了n_actions, visibility_type, 而没有probs_type, probs_relationship
                all_data.append({
                    'subject_id': task_df['participant_id'].iloc[0],
                    'task_id': f"task_{task_df['task_number'].unique()[0]}_run_{task}",
                    'task_type': task_df['visibility_type'].unique()[0] if 'visibility_type' in task_df.columns else 'unknown',
                    'n_actions': n_actions,
                    'actions': choices,
                    'rewards': rewards,
                    'best_actions': best_arm,   # actions with largest rewards for each states along the time
                    'action_reward_probs': mean_reward_by_action,   # 满分100
                    'X': X,
                    'Y': choices
                })

        return pd.DataFrame(all_data)

    def get_long_format_data(self) -> pd.DataFrame:
        """
        Explodes the nested session-level dataframe into a trial-by-trial long format
        for plotting and classical analyses.
        """
        records = []
        for _, row in self.df.iterrows():
            n_trials = len(row['actions'])
            for t in range(n_trials):
                records.append({
                    'subject_id': row['subject_id'],
                    'task_id': row['task_id'],
                    'task_type': row['task_type'],
                    'trial_number': t + 1,
                    'action_chosen': row['actions'][t],
                    'reward_received': row['rewards'][t],
                    'true_best_action': row['best_actions'][t],
                    'true_arm_probs': row['action_reward_probs']
                })
        return pd.DataFrame(records)

    def _get_long_format_data(self) -> pd.DataFrame:
        """Backward-compatible alias used by older notebook cells."""
        return self.get_long_format_data()

    # --- DATA EXTRACTION INTERFACES ---

    def get_dataset_description(self) -> dict:
        """
        Provides a summary of the dataset, including counts of unique subjects, tasks, and trials.
        """
        return {
            "n_subjects": self.df['subject_id'].nunique(),
            "n_tasks": self.df['task_id'].nunique(),
            "n_trials": int(self.df['actions'].apply(len).sum()),
            "task_ids": self.df['task_id'].unique().tolist(),
            "task_types": self.df['task_type'].unique().tolist(),
            "subject_ids": self.df['subject_id'].unique().tolist()
        }

    def get_task_properties(self, task_id: str) -> dict:
        """
        Extracts task design information from the data metadata.
        """
        task_df = self.df[self.df['task_id'] == task_id]
        if task_df.empty:
            raise ValueError(f"Task ID '{task_id}' not found in dataset.")

        action_lengths = task_df['actions'].apply(len)
        n_trials = int(action_lengths.max()) if not action_lengths.empty else 0
        n_actions = int(task_df['n_actions'].iloc[0])
        reward_sums = np.zeros(n_actions, dtype=float)
        reward_counts = np.zeros(n_actions, dtype=float)
        for idx in range(len(task_df)):
            seq = np.asarray(task_df.iloc[idx]['actions'], dtype=int)
            rewards = np.asarray(task_df.iloc[idx]['rewards'], dtype=float)
            valid = seq >= 0
            if np.any(valid):
                seq = seq[valid]
                rewards = rewards[valid]
            for action in np.unique(seq):
                if 0 <= action < n_actions:
                    action_mask = seq == action
                    reward_sums[action] += np.sum(rewards[action_mask])
                    reward_counts[action] += np.sum(action_mask)

        # average reward per action across ALL sessions of this task (not just the last one)
        action_probs = np.divide(reward_sums, reward_counts, out=np.zeros(n_actions, dtype=float), where=reward_counts > 0)

        return {
            "task_type": task_df['task_type'].iloc[0],
            "n_actions": n_actions,
            "n_trials": n_trials,
            "action_reward_probs": action_probs,
        }

    def get_human_data(self, task_id: str, subject_id: str = None) -> tuple:
        """Backward-compatible alias for task-level choice/reward sequences."""
        return self.get_sequential_data(task_id=task_id, subject_id=subject_id)

    def get_sequential_data(self, task_id: str, subject_id: str = None) -> tuple:
        """
        Extracts 1D sequences of choices and rewards for classical loop-based models
        like Rescorla-Wagner.
        """
        task_df = self.df[self.df['task_id'] == task_id]
        if task_df.empty:
            raise ValueError(f"Task ID '{task_id}' not found.")

        if subject_id is not None:
            task_df = task_df[task_df['subject_id'] == subject_id]
            if task_df.empty:
                raise ValueError(f"Subject '{subject_id}' not found under task '{task_id}'.")

        actions_list = []
        rewards_list = []
        for _, row in task_df.iterrows():
            actions = np.asarray(row['actions'], dtype=np.int64)
            rewards = np.asarray(row['rewards'], dtype=float)
            valid = actions >= 0
            if np.any(~valid):
                actions = actions[valid]
                rewards = rewards[valid]
            if actions.size > 0:
                actions_list.append(actions)
                rewards_list.append(rewards)
        return actions_list, rewards_list

    def get_tensor_data(self, tasks_subset: list = None, test_percentage: float = 0.2) -> tuple:
        """
        Extracts, stacks, and formats JAX 3D Tensors intended for Recurrent Neural Networks.
        Drops the problematic first trial of each session to ensure valid prediction.

        Variable-length sessions are truncated to the shortest session length in the subset
        before stacking so that all arrays share a common shape. This makes the workshop
        RNN workflow robust even when the dataset contains sessions with slightly different
        numbers of trials.

        For very small subsets (for example a single task), this method falls back to a
        same-data train/test split rather than raising a confusing error. This preserves
        notebook usability while still signaling that a proper generalization check should
        be run on a larger sample.
        """
        df = self.df if tasks_subset is None else self.df[self.df['task_id'].isin(tasks_subset)]
        if len(df) == 0:
            raise ValueError("The requested subset contains no sessions.")

        def _prepare_session_sequence(series_row):
            x = np.asarray(series_row['X'], dtype=float)
            y = np.asarray(series_row['Y'], dtype=int)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if y.ndim == 0:
                y = y.reshape(1)
            return x, y

        session_arrays = [_prepare_session_sequence(row) for _, row in df.iterrows()]
        session_lengths = [x.shape[0] for x, _ in session_arrays]
        target_len = min(session_lengths)

        X_sessions = []
        Y_sessions = []
        for x, y in session_arrays:
            X_sessions.append(x[:target_len])
            Y_sessions.append(y[:target_len])

        if len(df) == 1:
            X_single = X_sessions[0]
            Y_single = Y_sessions[0]
            X_train = X_single[1:] if X_single.ndim > 1 and X_single.shape[0] > 1 else X_single
            Y_train = Y_single[1:] if Y_single.ndim > 0 and Y_single.size > 1 else Y_single
            X_train_jax = jnp.array(X_train)
            Y_train_jax = jnp.array(Y_train)
            return X_train_jax, X_train_jax, Y_train_jax, Y_train_jax

        random_indices = np.random.permutation(len(df))
        test_size = max(1, int(len(df) * test_percentage)) if len(df) > 1 else 0
        test_indices = random_indices[:test_size]
        train_indices = random_indices[test_size:]

        if len(train_indices) == 0 or len(test_indices) == 0:
            train_indices = random_indices
            test_indices = random_indices

        X_train_stacked = np.stack([X_sessions[i] for i in train_indices])
        Y_train_stacked = np.stack([Y_sessions[i] for i in train_indices])
        X_test_stacked = np.stack([X_sessions[i] for i in test_indices])
        Y_test_stacked = np.stack([Y_sessions[i] for i in test_indices])

        X_train_jax = jnp.array(X_train_stacked[:, 1:, :])
        Y_train_jax = jnp.array(Y_train_stacked[:, 1:])
        X_test_jax = jnp.array(X_test_stacked[:, 1:, :])
        Y_test_jax = jnp.array(Y_test_stacked[:, 1:])

        return X_train_jax, X_test_jax, Y_train_jax, Y_test_jax

    def get_subject_task_dict(self):
        """
		Returns a dictionary mapping each subject to their respective tasks.
		"""
        subject_task_dict = {}
        for subject in self.df['subject_id'].unique():
            tasks = self.df[self.df['subject_id'] == subject]['task_id'].unique().tolist()
            subject_task_dict[subject] = tasks
        return subject_task_dict

    def get_task_subject_dict(self):
        """
		Returns a dictionary mapping each task to their respective subjects.
		"""
        task_subject_dict = {}
        for task in self.df['task_id'].unique():
            subjects = self.df[self.df['task_id'] == task]['subject_id'].unique().tolist()
            task_subject_dict[task] = subjects
        return task_subject_dict

    # --- PLOTTING & METRICS HELPERS ---

    def _add_wsls_metrics(self, df):
        """Internal helper to calculate Win-Stay/Lose-Shift metrics on long format data."""
        df['prev_action'] = df.groupby(['subject_id', 'task_id'])['action_chosen'].shift(1)
        df['prev_reward'] = df.groupby(['subject_id', 'task_id'])['reward_received'].shift(1)

        df['stayed'] = (df['action_chosen'] == df['prev_action'])
        df['shifted'] = (~df['stayed']) & df['prev_action'].notna()
        return df

    def pull_average_learning_curve(self, learning_curves_list):
        """Pads and averages learning curves of variable lengths."""
        max_len = max(len(curve) for curves in learning_curves_list for curve in curves)
        padded_curves = [list(curve) + [np.nan] * (max_len - len(curve))
                         for curves in learning_curves_list for curve in curves]

        arr = np.array(padded_curves)
        return np.nanmean(arr, axis=0), scipy.stats.sem(arr, axis=0, nan_policy='omit')

    def plot_reward_rate_by_task(self, task_id, window=10, subjects_subset=None):
        """Plots rolling average reward over time across different tasks and subjects."""
        df = self._get_long_format_data()

        # Filter by tasks first to know which subjects are actually available
        df = df[df['task_id'] == task_id]

        available_subjects = df['subject_id'].unique()

        # Handle subjects_subset logic
        if not subjects_subset:  # Catches None or []
            subjects_subset = available_subjects
        else:
            missing_subjects = [sub for sub in subjects_subset if sub not in available_subjects]
            if missing_subjects:
                print(f"Warning: The following subjects are not available for the requested tasks: {missing_subjects}")

            # Keep only the subjects that actually exist in the data
            subjects_subset = [sub for sub in subjects_subset if sub in available_subjects]
            df = df[df['subject_id'].isin(subjects_subset)]

        if df.empty:
            print("No data available to plot.")
            return

        # Calculate rolling reward
        df['rolling_reward'] = df.groupby(['subject_id'])['reward_received']\
                                 .transform(lambda x: x.rolling(window, min_periods=1).mean())

        plt.figure(figsize=(10, 6))

        # Seaborn automatically handles multiple lines if you pass the whole dataframe and use 'hue'
        sns.lineplot(data=df, x='trial_number', y='rolling_reward', hue='subject_id', alpha=0.8)

        plt.title('Learning Curves: Rolling Reward Rate')
        plt.xlabel('Trial Number')
        plt.ylabel(f'Rolling Average Reward ({window} trials)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_subject_action_sequence(self, subject_id: str, task_id: str):
        """Plots the sequence of actions and rewards for a single subject and task."""
        df = self._get_long_format_data()
        df = df[(df['subject_id'] == subject_id) & (df['task_id'] == task_id)]

        if df.empty:
            raise ValueError(f"No data found for subject '{subject_id}' in task '{task_id}'.")

        plt.figure(figsize=(12, 4))

        # 1. Connect the choices with a faint line to visualize the trajectory over time
        plt.plot(df['trial_number'], df['action_chosen'], color='gray', alpha=0.3, zorder=1)

        # 2. Highlight the objectively best action in the background
        if 'true_best_action' in df.columns:
            plt.plot(df['trial_number'], df['true_best_action'], color='gold', linestyle='--',
                     linewidth=2, alpha=0.6, label='True Best Action', zorder=0)

        # 3. Separate data by reward outcome for distinct styling (rewards may be points, not just 0/1)
        rewards = df[df['reward_received'] > 0]
        no_rewards = df[df['reward_received'] <= 0]

        plt.scatter(rewards['trial_number'], rewards['action_chosen'],
                    color='mediumseagreen', label='Reward', marker='o', s=60, alpha=0.9, zorder=2)
        plt.scatter(no_rewards['trial_number'], no_rewards['action_chosen'],
                    color='indianred', label='No Reward', marker='x', s=50, alpha=0.9, zorder=2)

        # Format the axes: Ensure ALL possible actions are shown on the y-axis,
        # even if the subject never selected them.
        max_action_idx = max(df['action_chosen'].max(), df.get('true_best_action', 0).max())
        plt.yticks(range(int(max_action_idx) + 1))

        plt.title(f"Action Sequence: Subject '{subject_id}' on Task '{task_id}'")
        plt.xlabel("Trial Number")
        plt.ylabel("Action Chosen")
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
