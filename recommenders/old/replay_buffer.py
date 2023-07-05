import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ReplayBuffer_ActTypes(Dataset):
    """
    Custom torch dataset that takes existing replay buffer file
    and samples from it during training.

    Pickle file with s list, a int, s' list.

    TODO: Currently reward is not expected. Think about it.

    num_items is without padding item. Take not actual number but end
    of numeration, so that indexing is the same.

    Returns:
    s, a, r, s', is_end
    """

    def __init__(
        self,
        dir,
        map_rew_cat_func,
        num_items,
        state_name="state",
        next_state_name="next_state",
        action_name="action",
        end_name="is_done",
        action_type_name="is_buy",
        state_len_name="len_state",
        true_next_state_len_name="len_next_states",
    ):
        super().__init__()
        self.dir = dir
        self.num_items = num_items

        # Names
        self.state_name = state_name  # s
        self.next_state_name = next_state_name  # s'
        self.action_name = action_name  # a
        self.action_type_name = action_type_name
        self.end_name = end_name

        # Save function to map reward category to reward
        # (assuming reward is categorical)
        self.map_rew_func = map_rew_cat_func

        # Load data
        replay_df = pd.read_pickle(dir)

        # Transform each information column to numpy array (indexing is way faster)
        self.states = np.array(replay_df[self.state_name].values.tolist())
        self.next_states = np.array(replay_df[self.next_state_name].values.tolist())
        self.actions = replay_df[action_name].to_numpy()
        self.action_type = replay_df[action_type_name].to_numpy()
        self.reward = map_rew_cat_func(self.action_type)
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
            self.action_type[idx],
            self.true_state_len[idx],
            self.true_next_state_len[idx],
            self.is_end[idx],
        )
