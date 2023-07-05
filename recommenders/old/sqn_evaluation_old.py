import torch
import numpy as np
from torch.utils.data import Dataset
from recommenders.evaluate.eval_protocol import get_preds, get_hits_for_batch
from recommenders.data_utils.preprocessing import preprocess_val_data_incl_act_rew

# Note: The steps taken here are done always when one evaluation is run. It just takes
# a few seconds but should be kept in mind.

# TODO: Make option to just comp. total. - Just have one act. in col?


class EvaluationDataset(Dataset):
    """
    Custom torch dataset that takes existing val/test file,
    transforms it and samples from it during training.

    Action types are differentiated here!

    pad_pos defines whether padding items are added at beginning or
    end of sequence.
    - "end": [1,2,3,999,999]
    - "beg": [999,999,1,2,3]

    File structure (df-file):
    - SessionID,
    - Action (= Item interacted),
    - ActionType bool (= Click/Buy/...)

    Transform to:
    - State: Groupby SessionID, get cumcount to know how many
             items/actions before as new col, apply get_state_col
             to group
    - Action: Just item interacted with
    - Reward: Map ActionType to reward based on a_to_r_dict


    Returns:
    s - (batch_size, len_state) Padded state,
    a - (bacth_size) Action,
    r - (batch_size) Reward,
    action_type (batch_size) Action Type, numerical
    true_state_len (batch_sizue) Length of unpadded sequence
    """

    def __init__(
        self,
        dir,
        padding_id,
        state_len,
        action_to_reward_dict,
        pad_pos="end",
        action_type_name="is_buy",
        session_id_name="session_id",
        action_name="item_id",
    ):
        super().__init__()

        self.pad_id = padding_id
        self.pad_pos = pad_pos
        self.state_len = state_len
        self.a_to_r = action_to_reward_dict

        # Preprocess raw val data and return pd-DataFrame
        eval_df = preprocess_val_data_incl_act_rew(
            dir,
            padding_id=padding_id,
            pad_pos=pad_pos,
            state_len=state_len,
            action_to_reward_dict=action_to_reward_dict,
            action_type_name=action_type_name,
            action_name=action_name,
            session_id_name=session_id_name,
        )

        # Transform each information column to numpy array (indexing is way faster)
        self.states = np.array(eval_df["state"].values.tolist())
        self.actions = eval_df["action"].to_numpy()
        self.reward = eval_df["reward"].to_numpy()
        self.action_type = eval_df["action_type"].to_numpy()
        self.true_state_len = eval_df["true_state_len"].to_numpy()

        # Free memory
        del eval_df

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.reward[idx],
            self.action_type[idx],
            self.true_state_len[idx],
        )


def create_metric_dicts(action_types, action_types_dict, top_k):
    """
    Create and intit the dicts to store metrics for k and
    action type.

    Returns one dict for hit count for each action type (incl. total),
    each val is np array with hr for k. Same for ndcg. Third dict to
    store total count of action types (and total actions).
    """
    # Init each action type with total hits per k array
    action_types_hit_dict = {
        action_types_dict[type]: np.zeros(len(top_k)) for type in action_types
    }
    action_types_hit_dict["total"] = np.zeros(len(top_k))

    # Init each action type with total hits per k array
    action_types_ndcg_dict = {
        action_types_dict[type]: np.zeros(len(top_k)) for type in action_types
    }
    action_types_ndcg_dict["total"] = np.zeros(len(top_k))

    # Init number of individual actions
    action_types_count = {action_types_dict[type]: 0 for type in action_types}
    action_types_count["total"] = 0

    return action_types_hit_dict, action_types_ndcg_dict, action_types_count


def evaluate(
    evaluation_data_loader,
    model,
    device,
    action_types,
    action_types_dict,
    loss_function,
    top_k=[5, 10, 20],
    head_idx=0,
):
    """
    Compute CrossEntrLoss + HR/NDCG for evaluation dataset for each k
    and each action category.

    action_types: list of different actions that metrics should
                  be calculated for. E.g. click/purchases/...

    action_types_dict: mapping from action int to string

    model: Not wrapper class but actual model.
    """

    model.eval()

    with torch.no_grad():
        # Initialize result dicts for hr/ndcg/action-type-count
        (
            action_types_hit_dict,
            action_types_ndcg_dict,
            action_types_count,
        ) = create_metric_dicts(action_types, action_types_dict, top_k)

        # Init CE-Loss
        loss = 0

        for n_batch, (s, a, r, a_type, s_len) in enumerate(evaluation_data_loader):
            # Only send s to GPU - other operations are numpy based (i.e. CPU).
            s = s.to(device)

            # Get predictions (should be on cpu afterwards)
            preds = get_preds(states=s, true_len=s_len, model=model, head_idx=head_idx)

            # Send preds back to CPU
            preds = preds.to("cpu")

            # Compute CrossEntropyLoss
            loss += loss_function(preds, a)

            # Assign hits to according action category
            for curr_type in action_types:
                # Get mask of where tensor a_type matches current action
                mask = a_type == curr_type

                # Jump to next action, if empty
                if mask.sum() == 0:
                    continue

                # Get number of hits in action-batch for each k
                # TODO: Disable k for multiple ks, mpre efficient
                total_hits_per_k, total_ndcg_per_k = get_hits_for_batch(
                    predictions=preds[mask], true_idx=a[mask], top_k=top_k
                )

                # Add k-hits numpy array to current array to accumulate over batches
                action_types_hit_dict[action_types_dict[curr_type]] += total_hits_per_k
                action_types_hit_dict["total"] += total_hits_per_k

                # Add k-ndcg numpy array to current array to accumulate over batches
                action_types_ndcg_dict[action_types_dict[curr_type]] += total_ndcg_per_k
                action_types_ndcg_dict["total"] += total_ndcg_per_k

                # Add number of examples to total count
                n_act = mask.sum()
                action_types_count[action_types_dict[curr_type]] += n_act
                action_types_count["total"] += n_act

        # Calculate hit ratio
        for key, val in action_types_hit_dict.items():
            action_types_hit_dict[key] = val / action_types_count[key]

        # Calculate ndcg ratio
        for key, val in action_types_ndcg_dict.items():
            action_types_ndcg_dict[key] = val / action_types_count[key]

        # Calculate average loss over epoch
        loss = loss / len(evaluation_data_loader)

    return loss, action_types_hit_dict, action_types_ndcg_dict


def update_train_metrics(
    s,
    a,
    a_type,
    s_len,
    model,
    device,
    action_types_hit_dict_train,
    action_types_ndcg_dict_train,
    action_types_count_train,
    action_types,
    action_types_dict,
    top_k=[5, 10, 20],
    head_idx=0,
):
    """
    Same as evaluate function, but just for one batch instead of
    whole dataloader.

    Updates the global metric dicts action_types_hit_dict,
    action_types_ndcg_dict and action_types_count.
    """

    model.eval()

    with torch.no_grad():
        # Only send s to GPU - other operations are numpy based (i.e. CPU).
        s = s.to(device)

        # TODO: Here is potential for opt - we already have preds in train_step
        # Get predictions (should be on cpu afterwards)
        preds = get_preds(states=s, true_len=s_len, model=model, head_idx=head_idx)

        # Send preds back to CPU
        preds = preds.to("cpu")

        # Assign hits to according action category
        for curr_type in action_types:
            # Get mask of where tensor a_type matches current action
            mask = a_type == curr_type

            # Jump to next action, if empty
            if mask.sum() == 0:
                continue

            # Get number of hits in action-batch for each k
            # TODO: Disable k for multiple ks, more efficient
            total_hits_per_k, total_ndcg_per_k = get_hits_for_batch(
                predictions=preds[mask], true_idx=a[mask], top_k=top_k
            )

            # Update dicts as in evaluate function

            # Add k-hits numpy array to current array to accumulate over batches
            action_types_hit_dict_train[
                action_types_dict[curr_type]
            ] += total_hits_per_k
            action_types_hit_dict_train["total"] += total_hits_per_k

            # Add k-ndcg numpy array to current array to accumulate over batches
            action_types_ndcg_dict_train[
                action_types_dict[curr_type]
            ] += total_ndcg_per_k
            action_types_ndcg_dict_train["total"] += total_ndcg_per_k

            # Add number of examples to total count
            n_act = mask.sum()
            action_types_count_train[action_types_dict[curr_type]] += n_act
            action_types_count_train["total"] += n_act

    return (
        action_types_hit_dict_train,
        action_types_ndcg_dict_train,
        action_types_count_train,
    )
