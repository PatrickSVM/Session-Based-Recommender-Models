import pytest
import torch
import numpy as np
import pandas as pd
from recommenders.data_utils.preprocessing import (
    get_state,
    get_state_col,
    action_to_reward,
)
from recommenders.old.sqn_evaluation_old import (
    create_metric_dicts,
    get_hits_for_batch,
)
from recommenders.old.sqn_evaluation_old import EvaluationDataset


@pytest.mark.parametrize(
    "input,expected",
    [
        (4, np.array([0, 1, 2, 3, 999])),
        (0, np.array([999, 999, 999, 999, 999])),
        (7, np.array([2, 3, 4, 5, 6])),
    ],
)
def test_get_state(input, expected):
    n_items_bef = input
    item_hist = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    res = get_state(
        n_items_bef, item_hist=item_hist, state_len=5, pad_id=999, pad_pos="end"
    )

    assert np.array_equal(res, expected)


def test_get_state_col():
    input_goup_df = pd.DataFrame(
        {"item_id": [10, 20, 30, 40, 50, 60, 70], "n_items_bef": [0, 1, 2, 3, 4, 5, 6]}
    )

    output = pd.DataFrame(
        {
            "State": [
                [999, 999, 999],
                [10, 999, 999],
                [10, 20, 999],
                [10, 20, 30],
                [20, 30, 40],
                [30, 40, 50],
                [40, 50, 60],
            ]
        }
    )

    res = get_state_col(group_df=input_goup_df, pad_id=999, state_len=3, pad_pos="end")

    ass = True
    for i in range(len(res)):
        if not (res.iloc[i] == output.iloc[i, 0]).all():
            ass = False

    assert ass


def test_action_to_reward():
    # For action 0 give reward 10, ...
    a_to_r_dict = {0: 10, 1: 100, 2: 200}
    test_df = pd.DataFrame(
        {"Action_type": [0, 0, 0, 1, 2, 2], "Other_col": [1, 2, 3, 4, 5, 6]}
    )

    # As in evaluation framework, apply along rows to col
    res = test_df.Action_type.apply(action_to_reward, 1, a_to_r_dict=a_to_r_dict)
    assert np.equal(res.values, [10, 10, 10, 100, 200, 200]).all()


def test_EvaluationDataset():
    # Test transformations for example df in test folder
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

    # Save groundtruth for r, a, act_type
    true_rews = [10, 10, 10, 50, 10, 10, 50, 10, 10, 10, 50, 50]
    true_act = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 100, 101]
    true_act_typ = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]

    # Padding = End of Sequence (inside class its numpy)
    load_examp_end_pad = EvaluationDataset(
        dir="test/data_transform_test.df",
        padding_id=999,
        state_len=3,
        action_to_reward_dict={0: 10, 1: 50},
        pad_pos="end",
        action_type_name="ActionType",
        session_id_name="SessionID",
        action_name="Item",
    )

    # Padding = Beginning of Sequence
    load_examp_beg_pad = EvaluationDataset(
        dir="test/data_transform_test.df",
        padding_id=999,
        state_len=3,
        action_to_reward_dict={0: 10, 1: 50},
        pad_pos="beg",
        action_type_name="ActionType",
        session_id_name="SessionID",
        action_name="Item",
    )

    states_bool = (
        np.equal(load_examp_end_pad.states, truth_end).all()
        & np.equal(load_examp_beg_pad.states, truth_beg).all()
    )

    # Check rewards
    rew_bool = np.equal(load_examp_end_pad.reward, true_rews).all()
    act_bool = np.equal(load_examp_end_pad.actions, true_act).all()
    act_type_bool = np.equal(load_examp_end_pad.action_type, true_act_typ).all()

    assert states_bool & rew_bool & act_bool & act_type_bool


def test_HR_NDCG_calc():
    # Replicate steps in update_train_metrics for one batch of data
    # without passing a model - just creating model output by hand.
    # Essentially tests behaviour of update_train_metrics, evaluate,
    # get_hits_for_batch functions. Indirect also check_existence_and_rank
    # and compare_elements.

    # Init
    action_types = [0, 1]
    action_types_dict = {0: "click", 1: "buy"}
    top_k = [1, 2, 10]
    (
        action_types_hit_dict,
        action_types_ndcg_dict,
        action_types_count,
    ) = create_metric_dicts(action_types, action_types_dict, top_k)

    # Set batch as a, a_type, pred (a as ground truth, pred as prediciton)
    a = torch.tensor([9, 0, 2, 1, 1, 1, 9, 0, 1])
    a_type = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])

    # Set preds: (batch, num_actions)
    # Hits for different k's, r=right, w=wrong
    # top 1: rrrwww rrw, top 2: rrrwrw rrr, top 10: rrrrrrrrr

    # Set predictions (representing output in logit form
    # before softmaxing - topk order is the same)
    preds = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 9 c
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # 0 c
            [1, 2, 100, 2, 2, 2, 2, 2, 2, 2],  # 2 c
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 9 c
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # 0 c
            [1, 1.5, 100, 2, 2, 2, 2, 2, 2, 2],  # 2 c
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 9 b
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # 0 b
            [1, 3, 100, 2, 2, 2, 2, 2, 2, 2],  # 2 b
        ]
    )

    # Clicks: 6, Buys: 3
    # Top 1 indexes: 9, 0, 2, 9, 0, 2, 9, 0, 2 RRRWWWW RRW
    # Top 2 indexes: [9,8]     R
    #                [0,1]     R
    #                [2,...]   R
    #                [9,8]     W                 |   R+ with rank 9 (for top10)
    #                [0,1]     R+ with rank 2
    #                [2,...]   W                 |  R+ with rank 9 (for top10)
    #                [9,8]     R
    #                [0,1]     R
    #                [2,1]     R+ with rank 2
    #
    # Top 10 indexes: RRRRRRRRRR

    # Expected results:
    #
    # HIT-RATIO: clicks: 0.5, 2/3, 1         | buys: 2/3, 1, 1           | total: 0.5556, 0.7778, 1.0000
    # NDCG:      clicks: 0.5, 0.6051, 0.7055 | buys: 2/3, 0.8767, 0.8767 | total: 0.5556, 0.6958, 0.7627
    #
    # NDCG treats worse rankings lower, so a mdoel with higher ranking true labels will be better.

    # Compute true values (for NDCG rank 1: 1/log2(1+1))
    top_10 = 1
    top1_HR_click = 3 / 6
    top2_HR_click = 4 / 6
    top1_NDCG_click = 3 / 6
    top2_NDCG_click = (3 * 1 + 1 / np.log2(2 + 1)) / 6
    top10_NDCG_click = (
        3 * 1 + 1 / np.log2(2 + 1) + 1 / np.log2(9 + 1) + 1 / np.log2(9 + 1)
    ) / 6
    HR_click = [top1_HR_click, top2_HR_click, top_10]
    NDCG_click = [top1_NDCG_click, top2_NDCG_click, top10_NDCG_click]

    top1_HR_buy = 2 / 3
    top2_HR_buy = 3 / 3
    top1_NDCG_buy = 2 / 3
    top2_NDCG_buy = (2 + 1 / np.log2(2 + 1)) / 3
    top10_NDCG_buy = (2 + 1 / np.log2(2 + 1)) / 3
    HR_buy = [top1_HR_buy, top2_HR_buy, top_10]
    NDCG_buy = [top1_NDCG_buy, top2_NDCG_buy, top10_NDCG_buy]

    for curr_type in action_types:
        mask = a_type == curr_type

        if mask.sum() == 0:
            continue

        total_hits_per_k, total_ndcg_per_k = get_hits_for_batch(
            predictions=preds[mask], true_idx=a[mask], top_k=top_k
        )

        action_types_hit_dict[action_types_dict[curr_type]] += total_hits_per_k
        action_types_hit_dict["total"] += total_hits_per_k

        action_types_ndcg_dict[action_types_dict[curr_type]] += total_ndcg_per_k
        action_types_ndcg_dict["total"] += total_ndcg_per_k

        n_act = mask.sum()
        action_types_count[action_types_dict[curr_type]] += n_act
        action_types_count["total"] += n_act

    for key, val in action_types_hit_dict.items():
        action_types_hit_dict[key] = val / action_types_count[key]

    for key, val in action_types_ndcg_dict.items():
        action_types_ndcg_dict[key] = val / action_types_count[key]

    # Check if theoretical calcs match function updates
    assert np.isclose(action_types_hit_dict["click"], HR_click).all()
    assert np.isclose(action_types_hit_dict["buy"], HR_buy).all()
    assert np.isclose(action_types_ndcg_dict["click"], NDCG_click).all()
    assert np.isclose(action_types_ndcg_dict["buy"], NDCG_buy).all()
