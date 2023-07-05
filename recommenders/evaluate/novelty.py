import torch
import numpy as np


def is_unpopular(array_element, unpopular_items):
    """
    Vectorized np func to check if element in list of itemIDs.
    """
    return array_element in unpopular_items


def get_batch_novelty_rewards(
    predictions,
    unpopular_items,
    reward=1,
    topk_to_consider=1,
):
    """
    Get cumulated novelty reward for a batch of predcitions. On CPU.
    In paper, only the top prediction is covered. This function expects an
    action/prediciton tensor of (batch_size, action_space) and computes the
    (average) novelty reward for the top topk_to_consider number of items.


    predictions: torch.tensor (batch_size, action_space)
    unpopular_items: list
    reward: reward for novelty
    topk_to_consider: int number of actions to consider, 1=paper

    Return np.array (batch_size) containing averge novelty reward for topk_to_consider top actions.
    """

    # TODO: Auslagern ###########################
    # Idx has size (batch, maxk) so top max_k preds
    _, action_idx = torch.topk(predictions, k=topk_to_consider)

    # Transform to numpy
    action_idx = action_idx.to("cpu").numpy()
    # TODO: Auslagern ###########################

    is_unpopular_vect = np.vectorize(is_unpopular, excluded=["unpopular_items"])
    # Get numpy array of novelty rewards (batch_size)
    nov_rew = is_unpopular_vect(action_idx, unpopular_items=unpopular_items)
    nov_rew = nov_rew.astype(int) * reward
    avg_nov_rew_per_row = np.mean(nov_rew, axis=1)

    return avg_nov_rew_per_row
