import numpy as np
import pandas as pd


def get_state(n_items_bef, item_hist, state_len, pad_id, pad_pos):
    """
    Function to retrieve column of states given a group n_items_bef column.

    Input is n_items_bef and item_hist history.

    To keep in line with the paper's implementation, the returned seqs are
    of form [oldest, ..., newest, pad, ..., pad] if pad_pos="end".

    For pad_pos="beg" we get [pad,...,pad, oldest, ..., newest]
    """

    # We have enough experience to fill state without padding
    if n_items_bef >= state_len:
        # Return state_len actions before (history - not including current action)
        return item_hist[n_items_bef - state_len : n_items_bef]
    else:  # Padding needed
        n_pad = state_len - n_items_bef
        pad_width = (0, n_pad) if pad_pos == "end" else (n_pad, 0)
        return np.pad(
            item_hist[:n_items_bef],
            pad_width=pad_width,
            mode="constant",
            constant_values=pad_id,
        )


def get_state_col(
    group_df,
    pad_id,
    pad_pos,
    state_len,
    item_before_col="n_items_bef",
    action_name="item_id",
):
    """
    Take group_df and return column with (padded) states.

    action_name: Column name for actions taken
    """
    item_hist = group_df[action_name].values

    session_states = group_df[item_before_col].apply(
        get_state,
        item_hist=item_hist,
        state_len=state_len,
        pad_id=pad_id,
        pad_pos=pad_pos,
    )
    return session_states


def action_to_reward(action_type, a_to_r_dict):
    """
    Map action type to reward value.
    """
    return a_to_r_dict[action_type]


def get_total_actions(df, action_type_col="is_buy"):
    """
    Count total number of actions per action type.
    """
    return df[action_type_col].value_counts()


def preprocess_val_data_incl_act_rew(
    dir,
    padding_id,
    state_len,
    action_to_reward_dict,
    pad_pos="end",
    action_type_name="is_buy",
    session_id_name="session_id",
    action_name="item_id",
):
    """
    dir: directory of raw data, with format sessionID, Action/Item, ActionType

    Returns:
    eval_df DataFrame with cols:
        - "state"
        - "action"
        - "reward"
        - "action_type"
        - "true_state_len"
    """

    eval_df = pd.read_pickle(dir)

    # Add column with number of items before in the session
    eval_df["n_items_bef"] = eval_df.groupby(session_id_name).cumcount()

    # Add column for (padded) state representation
    eval_df["state"] = eval_df.groupby(session_id_name, group_keys=False).apply(
        get_state_col,
        pad_id=padding_id,
        pad_pos=pad_pos,
        state_len=state_len,
        action_name=action_name,
    )

    # Copy n_items_bef col to create true_next_state_len
    eval_df["true_next_state_len"] = eval_df.n_items_bef.copy()

    # Return true_length as well for packed usage
    # To make it work, artificially set true_len of first to 1
    eval_df.loc[eval_df.n_items_bef == 0, "n_items_bef"] = 1

    # Set true_len to max state length if greater than that
    eval_df.loc[eval_df.n_items_bef > state_len, "n_items_bef"] = state_len

    # Add true_next_state_len column (add one to n_items_bef,
    # no change in first state needed here, its 1 now, just fix max)
    eval_df["true_next_state_len"] = eval_df["true_next_state_len"] + 1
    eval_df.loc[
        eval_df.true_next_state_len > state_len, "true_next_state_len"
    ] = state_len

    # Add reward column
    eval_df["reward"] = eval_df[action_type_name].apply(
        action_to_reward, 1, a_to_r_dict=action_to_reward_dict
    )

    # Rename action and action_type column
    eval_df.rename(
        columns={
            action_type_name: "action_type",
            action_name: "action",
            "n_items_bef": "true_state_len",
        },
        inplace=True,
    )
    eval_df = eval_df[["state", "action", "reward", "action_type", "true_state_len"]]

    return eval_df


def get_next_state(n_items_bef, item_hist, state_len, pad_id, pad_pos):
    """
    Function to retrieve column of next_states given a group n_items_bef column.

    Input is n_items_bef and item_hist history.

    To keep in line with the paper's implementation, the returned seqs are
    of form [oldest, ..., newest, pad, ..., pad] if pad_pos="end".

    For pad_pos="beg" we get [pad,...,pad, oldest, ..., newest]

    Efficiency potential: Combine it with state computation.
    """

    # We have enough experience to fill state without padding
    # + 1 since next_state will be items before and curr action
    if n_items_bef + 1 >= state_len:
        # Return state_len actions before (history - not including current action)
        return item_hist[n_items_bef - state_len + 1 : n_items_bef + 1]
    else:  # Padding needed
        n_pad = state_len - n_items_bef - 1
        pad_width = (0, n_pad) if pad_pos == "end" else (n_pad, 0)
        return np.pad(
            item_hist[: n_items_bef + 1],
            pad_width=pad_width,
            mode="constant",
            constant_values=pad_id,
        )


def get_next_state_col(
    group_df,
    pad_id,
    pad_pos,
    state_len,
    item_before_col="n_items_bef",
    action_name="item_id",
):
    """
    Take group_df and return column with (padded) states.

    action_name: Column name for actions taken
    """
    item_hist = group_df[action_name].values

    session_states = group_df[item_before_col].apply(
        get_next_state,
        item_hist=item_hist,
        state_len=state_len,
        pad_id=pad_id,
        pad_pos=pad_pos,
    )

    return session_states


def preprocess_train_data_incl_act_rew(
    dir,
    padding_id,
    state_len,
    incl_reward=False,
    action_to_reward_dict=None,
    pad_pos="end",
    action_type_name="is_buy",
    session_id_name="session_id",
    action_name="item_id",
    change_out_col_names={},
):
    """
    Same functionality as preprocess_val_data_incl_act_rew but for the
    training data the next state and whether the episode ended
    is also returned for training purposes which is irrelevant
    for testing.

    TODO: Is it really irrelevant? Should you test ending states?
    No you should not i feel like.
    Answ: No! We can sample up to the last action and get s, a, r, a_t
    this is the prediciton for the last aciton. There is no further a that
    is tested! So it does not bias the eval metric.
    """

    train_df = pd.read_pickle(dir)

    # Add column with number of items before in the session
    train_df["n_items_bef"] = train_df.groupby(session_id_name).cumcount()

    # Add column specifiying if its the last action in session
    # How: Get last n_items_bef per group in each row, compare row
    # to n_tems_bef, only True for last action of session
    train_df["is_end"] = (
        train_df.groupby(session_id_name)["n_items_bef"].transform("last")
        == train_df["n_items_bef"]
    )

    # Add column for (padded) state representation
    train_df["state"] = train_df.groupby(session_id_name, group_keys=False).apply(
        get_state_col,
        pad_id=padding_id,
        pad_pos=pad_pos,
        state_len=state_len,
        action_name=action_name,
    )

    # Add column for (padded) next state representation
    train_df["next_state"] = train_df.groupby(session_id_name, group_keys=False).apply(
        get_next_state_col,
        pad_id=padding_id,
        pad_pos=pad_pos,
        state_len=state_len,
        action_name=action_name,
    )

    # Copy n_items_bef col to create true_next_state_len
    train_df["true_next_state_len"] = train_df.n_items_bef.copy()

    # Return true_length as well for packed usage
    # To make it work, artificially set true_len of first to 1
    train_df.loc[train_df.n_items_bef == 0, "n_items_bef"] = 1

    # Set true_len to max state length if greater than that
    train_df.loc[train_df.n_items_bef > state_len, "n_items_bef"] = state_len

    # Add true_next_state_len column (add one to n_items_bef,
    # no change in first state needed here, its 1 now, just fix max)
    train_df["true_next_state_len"] = train_df["true_next_state_len"] + 1
    train_df.loc[
        train_df.true_next_state_len > state_len, "true_next_state_len"
    ] = state_len

    if incl_reward:
        # Add reward column
        train_df["reward"] = train_df[action_type_name].apply(
            action_to_reward, 1, a_to_r_dict=action_to_reward_dict
        )

    # Rename action and action_type column
    train_df.rename(
        columns={
            action_type_name: "action_type",
            action_name: "action",
            "n_items_bef": "true_state_len",
        },
        inplace=True,
    )

    if incl_reward:
        train_df = train_df[
            [
                "state",
                "action",
                "reward",
                "next_state",
                "action_type",
                "true_state_len",
                "true_next_state_len",
                "is_end",
            ]
        ]
    else:
        train_df = train_df[
            [
                "state",
                "action",
                "next_state",
                "action_type",
                "true_state_len",
                "true_next_state_len",
                "is_end",
            ]
        ]

    # Change output col names in case its specified
    train_df.rename(columns=change_out_col_names, inplace=True)

    return train_df


def preprocess_val_data(
    dir,
    padding_id,
    state_len,
    pad_pos="end",
    session_id_name="session_id",
    action_name="item_id",
):
    """
    Function to prepare validation data for SMORL-evaluation protocol. Excludes action_types,
    includes r_action (offline), r_nov/r_div are online.

    dir: directory of raw data, with format sessionID, Action/Item, ActionType, Reward

    Returns:
    eval_df DataFrame with cols:
        - "state"
        - "action"
        - "true_state_len"
    """

    eval_df = pd.read_pickle(dir)

    # Add column with number of items before in the session
    eval_df["n_items_bef"] = eval_df.groupby(session_id_name).cumcount()

    # Add column for (padded) state representation
    eval_df["state"] = eval_df.groupby(session_id_name, group_keys=False).apply(
        get_state_col,
        pad_id=padding_id,
        pad_pos=pad_pos,
        state_len=state_len,
        action_name=action_name,
    )

    # Copy n_items_bef col to create true_next_state_len
    eval_df["true_next_state_len"] = eval_df.n_items_bef.copy()

    # Return true_length as well for packed usage
    # To make it work, artificially set true_len of first to 1
    eval_df.loc[eval_df.n_items_bef == 0, "n_items_bef"] = 1

    # Set true_len to max state length if greater than that
    eval_df.loc[eval_df.n_items_bef > state_len, "n_items_bef"] = state_len

    # Add true_next_state_len column (add one to n_items_bef,
    # no change in first state needed here, its 1 now, just fix max)
    eval_df["true_next_state_len"] = eval_df["true_next_state_len"] + 1
    eval_df.loc[
        eval_df.true_next_state_len > state_len, "true_next_state_len"
    ] = state_len

    # Rename columns
    eval_df.rename(
        columns={
            action_name: "action",
            "n_items_bef": "true_state_len",
        },
        inplace=True,
    )
    eval_df = eval_df[["state", "action", "true_state_len"]]

    return eval_df


def preprocess_train_data(
    dir,
    padding_id,
    state_len,
    pad_pos="end",
    reward_name="reward",
    session_id_name="session_id",
    action_name="item_id",
    change_out_col_names={},
):
    """
    Same functionality as preprocess_val_data but for the
    training data the next state and whether the episode ended
    is also returned for training purposes which is irrelevant
    for testing.

    dir: directory of raw data, with format sessionID, Action/Item, ActionType, Reward

    Returns:
    train_df DataFrame with cols:
        - "state"
        - "action"
        - "r_act"
        - "next_state"
        - "true_state_len"
        - "true_next_state_len"
        - "is_end"
    """

    # Only read if dir is path not dataframe
    if type(dir) == str:
        train_df = pd.read_pickle(dir)
    else:
        train_df = dir.copy()

    # Add column with number of items before in the session
    train_df["n_items_bef"] = train_df.groupby(session_id_name).cumcount()

    # Add column specifiying if its the last action in session
    # How: Get last n_items_bef per group in each row, compare row
    # to n_items_bef, only True for last action of session
    train_df["is_end"] = (
        train_df.groupby(session_id_name)["n_items_bef"].transform("last")
        == train_df["n_items_bef"]
    )

    # Add column for (padded) state representation
    train_df["state"] = train_df.groupby(session_id_name, group_keys=False).apply(
        get_state_col,
        pad_id=padding_id,
        pad_pos=pad_pos,
        state_len=state_len,
        action_name=action_name,
    )

    # Add column for (padded) next state representation
    train_df["next_state"] = train_df.groupby(session_id_name, group_keys=False).apply(
        get_next_state_col,
        pad_id=padding_id,
        pad_pos=pad_pos,
        state_len=state_len,
        action_name=action_name,
    )

    # Copy n_items_bef col to create true_next_state_len
    train_df["true_next_state_len"] = train_df.n_items_bef.copy()

    # Return true_length as well for packed usage
    # To make it work, artificially set true_len of first to 1
    train_df.loc[train_df.n_items_bef == 0, "n_items_bef"] = 1

    # Set true_len to max state length if greater than that
    train_df.loc[train_df.n_items_bef > state_len, "n_items_bef"] = state_len

    # Add true_next_state_len column (add one to n_items_bef,
    # no change in first state needed here, its 1 now, just fix max)
    train_df["true_next_state_len"] = train_df["true_next_state_len"] + 1
    train_df.loc[
        train_df.true_next_state_len > state_len, "true_next_state_len"
    ] = state_len

    # Rename columns
    train_df.rename(
        columns={
            reward_name: "r_act",
            action_name: "action",
            "n_items_bef": "true_state_len",
        },
        inplace=True,
    )

    # Change output col names in case its specified
    train_df.rename(columns=change_out_col_names, inplace=True)

    return train_df[
        [
            "state",
            "action",
            "r_act",
            "next_state",
            "true_state_len",
            "true_next_state_len",
            "is_end",
        ]
    ]
