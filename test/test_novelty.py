import numpy as np
import torch
from recommenders.evaluate.novelty import get_batch_novelty_rewards


def test_get_batch_novelty_rewards():
    preds = torch.tensor([[100, 50, 0, 0, 0], [100, -10, 10, 0, 0]])
    unpopular = set([0, 1, 10, 11, 12, 13])

    # First reward for top-1: max is 0 in both so 1*rew, 1*rew - rew=2
    nov_rew_true_1k = np.array([2, 2])
    # Second reward for top-2: max is 0,1 = 1/2*4 and 0,2 so 1/2*(2+0)=1
    nov_rew_true_2k = np.array([2, 1])

    res_1k = get_batch_novelty_rewards(
        predictions=preds, unpopular_items=unpopular, reward=2
    )

    res_2k = get_batch_novelty_rewards(
        predictions=preds, unpopular_items=unpopular, reward=2, topk_to_consider=2
    )
    assert np.equal(nov_rew_true_1k, res_1k).all()
    assert np.equal(nov_rew_true_2k, res_2k).all()
