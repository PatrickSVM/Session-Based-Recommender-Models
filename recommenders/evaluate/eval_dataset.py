import numpy as np
from torch.utils.data import Dataset
from recommenders.data_utils.preprocessing import preprocess_val_data


class EvaluationDataset(Dataset):
    """
    Custom torch dataset that takes existing val/test file,
    transforms it and samples from it during training.

    Action types will be ignored here!

    pad_pos defines whether padding items are added at beginning or
    end of sequence.
    - "end": [1,2,3,999,999]
    - "beg": [999,999,1,2,3]

    File structure (df-file):
    - SessionID,
    - Action (= Item interacted),

    Transform to:
    - state: Groupby SessionID, get cumcount to know how many
             items/actions before as new col, apply get_state_col
             to group
    - action: Just item interacted with
    - true_state_len: True length of each seuqence/state


    Returns:
    s - (batch_size, len_state) Padded state,
    a - (bacth_size) Action,
    true_state_len (batch_sizue) Length of unpadded sequence
    """

    def __init__(
        self,
        dir,
        padding_id,
        state_len,
        pad_pos="end",
        session_id_name="session_id",
        action_name="item_id",
    ):
        super().__init__()

        self.pad_id = padding_id
        self.pad_pos = pad_pos
        self.state_len = state_len

        # Preprocess raw val data and return pd-DataFrame
        eval_df = preprocess_val_data(
            dir,
            padding_id=padding_id,
            pad_pos=pad_pos,
            state_len=state_len,
            action_name=action_name,
            session_id_name=session_id_name,
        )

        # Transform each information column to numpy array (indexing is way faster)
        self.states = np.array(eval_df["state"].values.tolist())
        self.actions = eval_df["action"].to_numpy()
        self.true_state_len = eval_df["true_state_len"].to_numpy()

        # Free memory
        del eval_df

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.true_state_len[idx],
        )
