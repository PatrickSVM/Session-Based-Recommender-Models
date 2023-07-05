import torch
from recommenders.evaluate.coverage import (
    get_coverage_k,
    update_actions_covered_topk_dict,
)


def test_get_coverage():
    unique_top20 = set([1, 2, 10, 20, 30, 40])
    unpop = set([1, 2, 3, 4, 5])
    item_num = 10

    true_all_cov = 6 / 10
    true_unpop_cov = 2 / 5

    unpop_cov, all_cov = get_coverage_k(unique_top20, unpop, num_actions=item_num)

    assert true_all_cov == all_cov
    assert true_unpop_cov == unpop_cov


def test_update_actions_covered_topk_dict():
    preds = torch.tensor(
        [
            [10, 9, 8, 7, 6],  # k1: 0, k2: 0,1
            [0, 10, 9, 8, 7],  # k1: 1, k2: 1,2
            [1, 9, 8, 7, 11],
        ]
    )  # k1: 4, k2: 4,1
    act_dict = {1: set([0, 101, 102, 103]), 2: set([0, 1, 4, 101, 102, 103])}

    true_res = {1: set([0, 1, 4, 101, 102, 103]), 2: set([0, 1, 4, 2, 101, 102, 103])}

    act_dict = update_actions_covered_topk_dict(
        predictions=preds, actions_covered_topk_dict=act_dict, top_k=[1, 2]
    )

    assert true_res[1] == act_dict[1]
    assert true_res[2] == act_dict[2]
