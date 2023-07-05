import os
import pandas as pd
import numpy as np
from more_itertools import chunked
from recommenders.data_utils.preprocessing import get_state_col, get_next_state_col
from recommenders.ikea.data_utils.utils import load_json_to_list
from pandarallel import pandarallel
from recommenders.utils.tokenizer import Tokenizer
from recommenders.ikea.data_utils.gfile import Gfile


def prepare_sessions(info_list, session_prefix, to_drop=["propensity"]):
    """
    Takes in raw list of dicts from json file loading and returns
    df which is in the right format for preprocessing.
    """
    # Load into exploded dataframe
    df = pd.json_normalize(
        info_list,
        record_path="events",
        meta=["market", "fullVisitorId", "start_time_ms"],
        meta_prefix="Session_",
    )

    # Rename start time col for session (otherwise same as event time) and remove prefix for rest
    df.rename(columns={"Session_start_time_ms": "sessionStartTime"}, inplace=True)
    df.columns = [col.replace("Session_", "") for col in df.columns]

    # Drop cols if they exist
    for col in to_drop:
        if col in df.columns:
            df = df.drop(to_drop, axis=1)

    # Transform to session ids (each session number from 0 to num sessions)
    df = df.reset_index(drop=True)
    df["sessionID"] = (
        df.reset_index()
        .groupby(["market", "fullVisitorId", "sessionStartTime"])
        .ngroup()
    )

    # Add prefix to sess_id
    df.sessionID = str(session_prefix) + df.sessionID.astype(str)

    # Rename action to action_type
    df.rename(columns={"action": "action_type"}, inplace=True)

    return df


def add_reward_simple(session_df, action_to_rew_dict):
    """
    Takes session_df as input and adds reward column according to
    provided dict. If value not in dict mapped to 0.

    Note: The reward will anyways only be important for the inspirational
    clicks since these are the only actions in the RB for training the agent.

    TODO: Consider putting higher weight when it came from the feed vs. pip!
    """
    # Add reward column
    session_df["reward"] = session_df.action_type.map(
        action_to_rew_dict,
    )

    # Set all others to 0
    session_df["reward"] = session_df.reward.fillna(0)
    return session_df


def prepare_for_replay_buffer(
    df,
    to_drop=[
        "fullVisitorId",
        "sessionStartTime",
        "start_time_ms",
        "timestamp_ms",
        "visitNumber",
        "feed_location",
    ],
):
    """
    Takes session df and prepare it for being transformed
    into a replay buffer. Applicable for train and val/test.
    """
    # Drop all duplicates before dropping other columns
    df = df.drop_duplicates(keep="first")

    # Drop unnecessary cols
    df = df.drop(columns=to_drop)

    # Drop nans
    df = df.dropna()

    df = df[["sessionID", "item_id", "action_type", "market", "reward"]]

    return df


def prepare_full_data(
    gfile_example_path,
    action_to_rew_dict,
    proj_name="PROJECT-NAME",
    max_files=None,
    use_disk=True,
    save=True,
):
    """
    Loop over json-zipped clickstream files in gcp bucket directory
    and prepare them for splitting into train/val/test.

    If use_disk=True, files are temporarily written to disk.
    """
    gcp_object = Gfile(path=gfile_example_path, proj_name=proj_name)

    # Get all files in dir
    all_files = gcp_object.list_blobs()

    if all_files[0] == "":
        all_files = all_files[1:]

    full_df = pd.DataFrame()
    if use_disk or save:
        if not os.path.exists("./temp_data"):
            os.makedirs("./temp_data")
        if os.path.exists("./temp_data/temp_df.csv"):
            os.remove("./temp_data/temp_df.csv")
            print("Old file removed.\n")

    for i, file in enumerate(all_files):
        content_list = load_json_to_list(file_name=file, gfile_obj=gcp_object)
        df = prepare_sessions(info_list=content_list, session_prefix=f"{i}_")
        df = add_reward_simple(
            df,
            action_to_rew_dict=action_to_rew_dict,
        )
        df = prepare_for_replay_buffer(df)

        # Remove all rows where item_id is empty
        df = df[~(df.item_id == "")]

        # Remove all rows where item id has multiple entries (11,20,13)
        mask = df.item_id.str.match(r"\d+,.*")
        df = df[~mask]
        print(f"Removed {mask.sum()} rows with mutltiple item_ids.\n")

        if use_disk:
            # Append to current csv
            header = True if i == 0 else False
            mode = "w" if i == 0 else "a"
            df.to_csv("./temp_data/temp_df.csv", header=header, index=False, mode=mode)

            # Free memory
            del df

        else:
            # Concat to full df
            full_df = pd.concat([full_df, df], axis=0)

            # Free old memory
            del df

        print(f"File #{i} successfully added.")

        # Stop at max files
        if i + 1 == max_files:
            break

    if use_disk:
        full_df = pd.read_csv("./temp_data/temp_df.csv")

        if not save:
            os.rmdir("./temp_data/temp_df.csv")
            print("\nFile removed from memory.")

    if save:
        full_df.to_csv("./temp_data/temp_df.csv", header=True, index=False)
        print("Done - File written to disk.")

    # Drop index
    full_df.reset_index(inplace=True, drop=True)

    return full_df


def mark_last_inspiration_click(group):
    """
    Function to apply to grouped df (by sessionID) to mark
    the row with the last inspiraitonal click for this specific
    session.

    Otherwise there will never be an ending signal, since the last
    action in the data is a item view most of the time, which will
    get removed by preprocessing -> is_end = 0 everywhere.
    """
    # Find index of last inspiraitonal click
    cond = (group["action_type"] == "click_inspiration") | (
        group["action_type"] == "select_content"
    )
    last_insp_idx = group[cond].index.max()
    if pd.isna(last_insp_idx):
        # No inspirational clicks in this group
        return pd.Series([False] * len(group), index=group.index)
    else:
        # Set last_purchase for this row to True if it corresponds to the last purchase action
        last_insp_mask = group.index == last_insp_idx

        return pd.Series(last_insp_mask, index=group.index)


def get_next_state_with_future(df, future_steps_next_state):
    """
    Takes in grouped dataframe by sessionID and returns new next state col
    for the relvant actions (click_inspiration and content selection) because
    they will be the only one remaining for the replay buffer.

    Default before is just add the action to the sequence of clicks. Other possibility
    is to incorporate all interactions until next click (user as environment - product views
    are reactions of environment on image recommendation). This option is achieved by setting
    future_steps_next_state to "all".

    For only one or multiple more steps set future_steps_next_state > 1. 2 means that the action
    plus the interaction afterwards will be taken into account as well, etc.
    """
    if future_steps_next_state == "all":
        new_col = df[
            (df["action_type"] == "click_inspiration")
            | (df["action_type"] == "select_content")
        ].state.shift(-1)
        new_col.iloc[-1] = df[
            (df["action_type"] == "click_inspiration")
            | (df["action_type"] == "select_content")
        ].next_state.iloc[-1]

    elif future_steps_next_state > 1:
        new_col = df.state.shift(-future_steps_next_state)
        new_col.fillna(df.next_state, inplace=True)
        new_col = new_col[
            (df["action_type"] == "click_inspiration")
            | (df["action_type"] == "select_content")
        ]

    return new_col


def get_cum_rew(group_df, future_steps_next_state, reward_name):
    """
    Function that depending on future_steps_next_state computes the
    reward for the inspirational feed actions. If it is set to "all",
    all future rewards up to the end of the session or the next image
    click are summed up.

    For 2, only the immediate next one is summed up.
    """
    df = group_df.copy()
    df.reset_index(drop=True, inplace=True)

    if future_steps_next_state == "all":
        cond = (df["action_type"] == "click_inspiration") | (
            df["action_type"] == "select_content"
        )
        df["buy_group"] = (cond).cumsum()
        df["reward_sum"] = df.groupby("buy_group")[reward_name].transform("sum")
        # Create a shifted column to identify the 'buy' followed by 'buy' case
        df["next_action_type"] = df["action_type"].shift(-1)
        cond_2 = (df["next_action_type"] == "click_inspiration") | (
            df["next_action_type"] == "select_content"
        )

        # Replace the 'reward_sum' for 'buy' followed by 'buy' with the current 'reward'
        df.loc[(cond) & (cond_2), "reward_sum"] = df["reward"]

        # Replace the 'reward_sum' for 'buy' not followed by 'buy' with NaN
        df.loc[(cond) & (~cond_2), "reward_sum"] = np.nan

        # Backfill the NaN values
        df["reward_sum"] = df["reward_sum"].bfill()

        df["reward_sum"].fillna(df[reward_name], inplace=True)
        df.drop(columns=["buy_group", "reward"], inplace=True)
        df.rename(columns={"reward_sum": reward_name}, inplace=True)

    elif future_steps_next_state == 2:
        new_rew = df[reward_name] + df[reward_name].shift(-1)
        new_rew.fillna(df[reward_name], inplace=True)
        df[reward_name] = new_rew

    else:
        raise NotImplementedError
    df.index = group_df.index
    return df


def preprocess_train_data(
    dir,
    padding_id,
    state_len,
    tokenizer_market,
    tokenizer_input,
    tokenizer_output,
    pad_pos="end",
    reward_name="reward",
    session_id_name="session_id",
    action_name="item_id",
    change_out_col_names={},
    parallel=False,
    progress_bar=False,
    future_steps_next_state=1,
):
    """
    IKEA version - returns action-type as well.
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
    if type(dir) == str:
        train_df = pd.read_csv(dir)
    else:
        train_df = dir.copy()

    # Init tokenizers
    if type(tokenizer_input) == str:
        tokenizer_input = Tokenizer.from_file(tokenizer_input)

    if type(tokenizer_output) == str:
        tokenizer_output = Tokenizer.from_file(tokenizer_output)

    if type(tokenizer_market) == str:
        tokenizer_market = Tokenizer.from_file(tokenizer_market)

    # Tokenize item_id (with input tokenizer) and market col
    train_df.item_id = train_df.item_id.apply(
        lambda string: tokenizer_input.stoi(string)
    )
    train_df.market = train_df.market.apply(
        lambda string: tokenizer_market.stoi(string)
    )

    train_df.reset_index(inplace=True, drop=True)

    # Add column with number of items before in the session
    train_df["n_items_bef"] = train_df.groupby(session_id_name).cumcount()

    # Add column specifiying if its the last action in session
    # How: Get last n_items_bef per group in each row, compare row
    # to n_items_bef, only True for last action of session
    train_df["is_end"] = train_df.groupby("sessionID", group_keys=False).apply(
        mark_last_inspiration_click
    )

    if parallel:
        # Init pandarallel
        pandarallel.initialize(progress_bar=progress_bar)

        # Add column for (padded) state representation
        train_df["state"] = train_df.groupby(
            session_id_name, group_keys=False
        ).parallel_apply(
            get_state_col,
            pad_id=padding_id,
            pad_pos=pad_pos,
            state_len=state_len,
            action_name=action_name,
        )

        # Add column for (padded) next state representation
        train_df["next_state"] = train_df.groupby(
            session_id_name, group_keys=False
        ).parallel_apply(
            get_next_state_col,
            pad_id=padding_id,
            pad_pos=pad_pos,
            state_len=state_len,
            action_name=action_name,
        )
    else:
        # Add column for (padded) state representation
        train_df["state"] = train_df.groupby(session_id_name, group_keys=False).apply(
            get_state_col,
            pad_id=padding_id,
            pad_pos=pad_pos,
            state_len=state_len,
            action_name=action_name,
        )

        # Add column for (padded) next state representation
        train_df["next_state"] = train_df.groupby(
            session_id_name, group_keys=False
        ).apply(
            get_next_state_col,
            pad_id=padding_id,
            pad_pos=pad_pos,
            state_len=state_len,
            action_name=action_name,
        )

    train_df.reset_index(inplace=True, drop=True)

    # Get the next state column and cumulated reward defined by future_steps_next_state
    if future_steps_next_state != 1:
        # New next state col
        train_df.loc[
            (train_df["action_type"] == "click_inspiration")
            | (train_df["action_type"] == "select_content"),
            "next_state",
        ] = train_df.groupby("sessionID", group_keys=False).apply(
            get_next_state_with_future, future_steps_next_state=future_steps_next_state
        )

        # According cumulated reward signal
        train_df = train_df.groupby("sessionID", group_keys=False).apply(
            get_cum_rew,
            future_steps_next_state=future_steps_next_state,
            reward_name=reward_name,
        )

    # Remove all rows where the action_type!="click_inspirational"/"select_content"
    # Only real actions aka inspirational clicks will remain with the right states,
    # next states and rewards.
    train_df = train_df[
        (train_df["action_type"] == "click_inspiration")
        | (train_df["action_type"] == "select_content")
    ]

    # Now map the actions to the output tokens (doing it here is computationally advanteagous)

    # First: Map back to OG
    # Note: This is memory-efficient way is only possible if no inputs mapped to <unk>
    train_df[action_name] = train_df[action_name].apply(
        lambda index: tokenizer_input.itos(index)
    )

    # Second: Map to output
    train_df[action_name] = train_df[action_name].apply(
        lambda string: tokenizer_output.stoi(string)
    )

    # Copy n_items_bef col to create true_next_state_len
    train_df["true_state_len"] = train_df.state.apply(
        lambda state: state_len - np.count_nonzero(state == tokenizer_input.pad_idx)
    )

    # Return true_length as well for packed usage
    # To make it work, artificially set true_len of first to 1
    train_df.loc[train_df.true_state_len == 0, "true_state_len"] = 1

    # Copy n_items_bef col to create true_next_state_len
    train_df["true_next_state_len"] = train_df.next_state.apply(
        lambda state: state_len - np.count_nonzero(state == tokenizer_input.pad_idx)
    )

    # Rename columns
    train_df.rename(
        columns={
            reward_name: "r_act",
            action_name: "action",
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
