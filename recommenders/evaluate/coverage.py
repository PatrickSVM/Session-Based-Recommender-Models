import torch


def get_coverage_k(unique_topk_actions, unpopular_items, num_actions):
    """
    Compute coverage for a given set of unique top-k actions accumulated
    over the whole run over the dataset (i.e. one batch) - either during
    training or during evaluation).

    Important: dtype is set here.

    Returns coverage for unpopular and all items given top-k accumulated unique actions.
    """
    assert isinstance(unique_topk_actions, set) & isinstance(unpopular_items, set)

    # Intersect the two sets and return percentages
    intersect_unpop = unique_topk_actions.intersection(unpopular_items)
    unpopular_coverage = len(intersect_unpop) / len(unpopular_items)
    all_coverage = len(unique_topk_actions) / num_actions

    return unpopular_coverage, all_coverage


def update_actions_covered_topk_dict(
    predictions,
    actions_covered_topk_dict,
    top_k,
    input_tokenizer=None,
    output_tokenizer=None,
):
    """
    Takes batch of predictions and updates the set of unique actions, that were
    recommended by the model along the epoch for each k in topk. So basically
    take old set of actions so far and make union with batch_topk_predicitons.

    actions_covered_topk_dict: Dict that contains set of unique already covered
                               actions during epoch for each k
    """
    # Get max k
    max_k = max(top_k)

    # Idx has size (batch, maxk) so top max_k preds
    _, top_actions = torch.topk(predictions, k=max_k)

    for k in top_k:
        # Get topk actions as list and create union with old topk
        all_actions_k = top_actions[:, :k].flatten().tolist()
        updated_set = actions_covered_topk_dict[k].union(all_actions_k)

        # Update set
        actions_covered_topk_dict[k] = updated_set

    return actions_covered_topk_dict


def get_coverage(actions_covered_topk_dict, unpopular_set, num_actions, topk):
    """
    Takes batch of predictions and computes novelty- and all-item-coverage
    for each k in topk at the end of the episode, based on the accumulated
    set of topk-actions and the unpopular item/action set.

    actions_covered_topk_dict: Dict that contains set of already covered
                               actions during epoch for each k
    """
    coverage_res = {}
    for k in topk:
        # Get coverage - (unpopular_cov, all_item_cov)
        k_res = get_coverage_k(
            unique_topk_actions=actions_covered_topk_dict[k],
            unpopular_items=unpopular_set,
            num_actions=num_actions,
        )
        coverage_res[k] = k_res
    return coverage_res
