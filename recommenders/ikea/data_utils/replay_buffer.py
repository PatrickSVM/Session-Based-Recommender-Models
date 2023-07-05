import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    """
    IKEA version.

    Custom torch dataset that takes existing replay buffer file
    and samples from it during training.

    Pickle file with following cols:
    - s list,
    - a int,
    - r_act float,
    - s' list,
    - is_end,
    - state_len
    - next_state_len

    Returns:
    s, a, r, s', s_len, s_next_len, is_end
    """

    def __init__(
        self,
        dir,
        state_name="state",
        next_state_name="next_state",
        action_name="action",
        end_name="is_end",
        state_len_name="true_state_len",
        true_next_state_len_name="true_next_state_len",
        reward_name="r_act",
    ):
        super().__init__()
        self.dir = dir

        # Names
        self.state_name = state_name  # s
        self.next_state_name = next_state_name  # s'
        self.reward_name = reward_name  # r
        self.action_name = action_name  # a
        self.end_name = end_name

        # Load data directly from GCP bucket
        replay_df = pd.read_json(dir, orient="records", lines=True)

        # Transform each information column to numpy array (indexing is way faster)
        self.states = np.array(replay_df[state_name].values.tolist())
        self.next_states = np.array(replay_df[next_state_name].values.tolist())
        self.actions = replay_df[action_name].to_numpy()
        self.reward = replay_df[reward_name].to_numpy()
        self.true_state_len = replay_df[state_len_name].to_numpy()
        self.true_next_state_len = replay_df[true_next_state_len_name].to_numpy()
        self.is_end = replay_df[end_name].to_numpy()

        # Free memory
        del replay_df

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.reward[idx],
            self.next_states[idx],
            self.true_state_len[idx],
            self.true_next_state_len[idx],
            self.is_end[idx],
        )


class EvaluationDataset(Dataset):
    """
    Custom torch dataset that takes existing replay buffer file
    for evaluation and samples from it during training.

    Same behaviour as training buffer but only returns s, a, s_len
    for efficiency purposes.

    Pickle file with following cols:
    - s list,
    - a int,
    - state_len

    Returns:
    s, a, s_len
    """

    def __init__(
        self,
        dir,
        state_name="state",
        action_name="action",
        state_len_name="true_state_len",
    ):
        super().__init__()
        self.dir = dir

        # Names
        self.state_name = state_name  # s
        self.action_name = action_name  # a

        # Load data directly from GCP bucket
        replay_df = pd.read_json(dir, orient="records", lines=True)

        # Transform each information column to numpy array (indexing is way faster)
        self.states = np.array(replay_df[state_name].values.tolist())
        self.actions = replay_df[action_name].to_numpy()
        self.true_state_len = replay_df[state_len_name].to_numpy()

        # Free memory
        del replay_df

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.true_state_len[idx],
        )
