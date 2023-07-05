import torch
from recommenders.utils.tensor_operations import (
    gather_from_3d,
    weighted_q_loss,
    get_weighted_q_target,
    get_max_action,
)


def test_gather_from_3d():
    q_acc = torch.Tensor([[1, 2, 3, 4, 5], [-1, -2, -3, -4, -5]])
    q_div = torch.Tensor([[10, 20, 30, 40, 50], [0.1, 0.2, 0.3, 0.4, 0.5]])
    q_nov = torch.Tensor([[100, 200, 300, 400, 500], [11, 21, 31, 41, 51]])

    all_q = torch.stack([q_acc, q_div, q_nov], dim=1)  # (2, 3, 5)

    chosen_actions = torch.tensor([3, 1])  # Chosen action idx, sample 1:3, sample2:1
    true_res = torch.tensor([[4, 40, 400], [-2, 0.2, 21]])

    res = gather_from_3d(q_tensor=all_q, action_idxs=chosen_actions, num_heads=3)

    assert torch.equal(true_res, res)


def test_weighted_q_loss():
    q_vals = torch.tensor([[4, 40, 400], [-2, 0.2, 21]])
    w = torch.tensor([0.1, 0.5, 0.4])
    res = weighted_q_loss(q_vals, w)

    true_res = torch.tensor([180.4, 8.3])

    assert torch.equal(res, true_res)


def test_get_weighted_q_target():
    q_acc = torch.Tensor([[1, 2, 3], [-1, -2, -3]])
    q_div = torch.Tensor([[10, 20, 30], [0.1, 0.2, 0.3]])
    q_nov = torch.Tensor([[100, 200, 300], [11, 21, 31]])
    all_q = torch.stack([q_acc, q_div, q_nov], dim=1)  # (2, 3, 3)

    w = torch.tensor([0.1, 0.5, 0.4])

    res = get_weighted_q_target(all_q, w)

    true_res = torch.tensor(
        [
            [
                0.1 * 1 + 0.5 * 10 + 100 * 0.4,
                2 * 0.1 + 20 * 0.5 + 200 * 0.4,
                3 * 0.1 + 30 * 0.5 + 300 * 0.4,
            ],
            [
                0.1 * -1 + 0.5 * 0.10 + 11 * 0.4,
                -2 * 0.1 + 0.20 * 0.5 + 21 * 0.4,
                -3 * 0.1 + 0.30 * 0.5 + 31 * 0.4,
            ],
        ]
    )

    assert torch.allclose(res, true_res)


def test_get_max_action():
    q_acc = torch.Tensor([[1, 2, 3], [-1, -2, -3]])
    q_div = torch.Tensor([[10, 20, 30], [0.1, 0.2, 0.3]])
    q_nov = torch.Tensor([[100, 200, 300], [11, 21, 31]])
    all_q = torch.stack([q_acc, q_div, q_nov], dim=1)  # (2, 3, 3)

    w = torch.tensor([0.1, 0.5, 0.4])

    res = get_weighted_q_target(all_q, w)

    act_idxs = get_max_action(res)
    true_res = torch.tensor(
        [2, 2]
    )  # Index of max action per sample row in res form above

    assert torch.equal(act_idxs, true_res)
