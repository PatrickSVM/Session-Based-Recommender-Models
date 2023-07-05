import numpy as np
from recommenders.data_utils.preprocessing import preprocess_train_data_incl_act_rew


def test_preprocess_train_data_incl_act_rew():
    # Save groundtruth for both padding cases for states
    truth_end = np.array(
        [
            [999, 999, 999],
            [1, 999, 999],
            [1, 2, 999],
            [1, 2, 3],
            [999, 999, 999],
            [6, 999, 999],
            [6, 7, 999],
            [6, 7, 8],
            [7, 8, 9],
            [8, 9, 10],
            [999, 999, 999],
            [100, 999, 999],
        ]
    )

    truth_beg = np.array(
        [
            [999, 999, 999],
            [999, 999, 1],
            [999, 1, 2],
            [1, 2, 3],
            [999, 999, 999],
            [999, 999, 6],
            [999, 6, 7],
            [6, 7, 8],
            [7, 8, 9],
            [8, 9, 10],
            [999, 999, 999],
            [999, 999, 100],
        ]
    )

    # True next states
    s_next_end = np.array(
        [
            [1, 999, 999],
            [1, 2, 999],
            [1, 2, 3],
            [2, 3, 4],
            [6, 999, 999],
            [6, 7, 999],
            [6, 7, 8],
            [7, 8, 9],
            [8, 9, 10],
            [9, 10, 11],
            [100, 999, 999],
            [100, 101, 999],
        ]
    )

    s_next_beg = np.array(
        [
            [999, 999, 1],
            [999, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [999, 999, 6],
            [999, 6, 7],
            [6, 7, 8],
            [7, 8, 9],
            [8, 9, 10],
            [9, 10, 11],
            [999, 999, 100],
            [999, 100, 101],
        ]
    )

    # Save groundtruth for r, a, act_type
    true_rews = [10, 10, 10, 50, 10, 10, 50, 10, 10, 10, 50, 50]
    true_act = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 100, 101]
    true_act_typ = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]
    true_lens = [1, 1, 2, 3, 1, 1, 2, 3, 3, 3, 1, 1]
    true_next_lens = [1, 2, 3, 3, 1, 2, 3, 3, 3, 3, 1, 2]

    train_end_pad = preprocess_train_data_incl_act_rew(
        dir="./test/data_transform_test.df",
        padding_id=999,
        state_len=3,
        incl_reward=True,
        action_to_reward_dict={0: 10, 1: 50},
        pad_pos="end",
        action_type_name="ActionType",
        session_id_name="SessionID",
        action_name="Item",
    )

    train_beg_pad = preprocess_train_data_incl_act_rew(
        dir="./test/data_transform_test.df",
        padding_id=999,
        state_len=3,
        incl_reward=True,
        action_to_reward_dict={0: 10, 1: 50},
        pad_pos="beg",
        action_type_name="ActionType",
        session_id_name="SessionID",
        action_name="Item",
    )

    # Check states
    state_vals_end = np.array(train_end_pad["state"].values.tolist())
    state_vals_beg = np.array(train_beg_pad["state"].values.tolist())
    states_bool = (
        np.equal(state_vals_end, truth_end).all()
        & np.equal(state_vals_beg, truth_beg).all()
    )

    # Check next states
    next_state_vals_end = np.array(train_end_pad["next_state"].values.tolist())
    next_state_vals_beg = np.array(train_beg_pad["next_state"].values.tolist())
    states_next_bool = (
        np.equal(next_state_vals_end, s_next_end).all()
        & np.equal(next_state_vals_beg, s_next_beg).all()
    )

    # Check rewards
    rew_bool = np.equal(train_end_pad.reward, true_rews).all()
    act_bool = np.equal(train_beg_pad.action, true_act).all()
    act_type_bool = np.equal(train_beg_pad.action_type, true_act_typ).all()
    true_state_len_bool = np.equal(train_beg_pad.true_state_len, true_lens).all()
    true_next_state_len_bool = np.equal(
        train_beg_pad.true_next_state_len, true_next_lens
    ).all()

    assert (
        states_bool
        & states_next_bool
        & rew_bool
        & act_bool
        & act_type_bool
        & true_state_len_bool
        & true_next_state_len_bool
    )


# TODO: Test types of embedded seqs - packed or unpacked?
