import torch
import numpy as np
from recommenders.evaluate.coverage import (
    get_coverage,
    update_actions_covered_topk_dict,
)
from recommenders.evaluate.diversity import get_batch_diversity_rewards
from recommenders.evaluate.novelty import get_batch_novelty_rewards
from recommenders.evaluate.repetitiveness import get_batch_repetitions


@np.vectorize
def compare_elements(array_to_check, vals_to_find):
    """
    array_to_check contains rows with labels and
    check for each row if vals_to_find in this row
    is in the according row of array_to_check.

    Vectorizing will apply the function to each element in
    array_to_check and compare it to the according value in
    the row of vals_to_find.
    """
    return array_to_check == vals_to_find


def check_existence_and_rank(array_to_check, vals_to_find):
    """
    Apply the vectorized compare_element function and
    squeeze output for each row to True if one of the
    elements was the same as the value to compare, False
    respectively into a mask.

    Init result array with 0,
    0 means true value not included in top-k result.

    Apply mask to result array, if mask=True, value is
    included in top-k preds. Then get argmax of respective
    row, which produces index/rank, since max. one  True
    element per row.

    Note: If true label is top-1 pred, will also produce 0. That
    is why 1 is added to all indexes. This way 0:Not included, >=1:Rank.

    NDCG expects rank (=index+1) anyways.

    array_to_check: (batch_size, k) - Top-k prediction indexes (taken from topk applied on forward pass)
    vals_to_find: (batch_size) - True action taken by user/ True index of action/item in output vector of NN
    """

    # Get bool array (batch_size, k) for matching of values rowwise
    element_wise_comp_array = compare_elements(array_to_check, vals_to_find)

    # Get mask to check which rows contain true label
    mask = element_wise_comp_array.any(1)

    rank_result = np.zeros(shape=len(vals_to_find))
    rank_result[mask] = np.argmax(element_wise_comp_array[mask], axis=1) + 1

    return mask, rank_result


def get_hits_for_batch(predictions, true_idx, top_k=[5, 10, 20]):
    """
    Compute number of hits and sum of ndcg values
    for batch predictions and batch actions (true_idx).

    predictions: (batch_size, num_actions)
    true_idx: (batch_size)
    """

    # Get max k
    max_k = max(top_k)

    # Idx has size (batch, maxk) so top max_k preds
    _, idx = torch.topk(predictions, k=max_k)

    # Transform to numpy
    idx = idx.numpy()
    true_idx = true_idx.unsqueeze(1).numpy()

    hit_list = np.zeros(len(top_k))
    ndcg_list = np.zeros(len(top_k))

    for i, k in enumerate(top_k):
        # Check if true_idx is in top-k (:k!) predictions and get ranks
        target_in_k_pred, rank_k = check_existence_and_rank(idx[:, :k], true_idx)

        # Compute number of hits for top-k preds
        hits_k = target_in_k_pred.sum()

        # Compute NDCG of hits for top-k preds, assign 0 if rank=0, so no match (otherwise log(1)=0)
        denom = np.log2(rank_k + 1)
        ndcg_k = np.divide(
            target_in_k_pred, denom, out=np.zeros_like(denom), where=denom != 0
        ).sum()

        hit_list[i] = hits_k
        ndcg_list[i] = ndcg_k

    return hit_list, ndcg_list


def get_preds(states, true_len, model, head_idx):
    """
    Return predictions from model given batch-states.

    Head index is the index of the head that should be used.

    States: (batch_size, num_actions)
    """
    # Get tuple of all predicitons
    all_preds = model(states, true_len)

    # If tuple, retrieve the prediciton head
    if isinstance(all_preds, tuple):
        preds = all_preds[head_idx]
    else:
        preds = all_preds

    return preds


def evaluate(
    evaluation_data_loader,
    model,
    device,
    loss_function,
    padding_pos,
    diversity_embedding,
    unpopular_actions_set,
    head_idx=0,
    topk_hr_ndcg=[5, 10, 20],
    topk_to_consider_div=1,
    topk_to_consider_nov=1,
    topk_to_consider_cov=[1, 5, 10],
    novelty_rew_signal=1,
    input_tokenizer=None,
    output_tokenizer=None,
):
    """
    Compute
    - CrossEntrLoss
    - HR@k
    - NDCG@k
    - diversity as CV@k on all items
    - novelty as CV@k on less popular items
    - cum online novelty reward
    - cum online diversity reward
    for an evaluation dataset.


    model: Not wrapper class but actual model.
    unpopular_actions_set: UNpopular items/actions. This must be a SET.
    """

    model.eval()

    with torch.no_grad():
        # Initialize result arrays for total-hr/ndcg counts
        total_novelty_rew = 0
        total_diversity_rew = 0
        hr = np.zeros(shape=len(topk_hr_ndcg))
        ndcg = np.zeros(shape=len(topk_hr_ndcg))
        repetitions = np.zeros(shape=len(topk_hr_ndcg))
        n_total_samples = 0

        # Init dict of sets of covered actions for each k
        actions_covered_topk_dict = {k: set() for k in topk_to_consider_cov}

        # Init CE-Loss
        loss = 0

        for n_batch, (s, a, s_len) in enumerate(evaluation_data_loader):
            # Only send s/s_len/a to GPU - other operations are numpy based (i.e. CPU).
            s = s.to(device)
            a = a.to(device)

            # Get predictions (on GPU)
            preds = get_preds(states=s, true_len=s_len, model=model, head_idx=head_idx)

            # Compute CrossEntropyLoss
            loss += loss_function(preds, a)

            # Compute average diversity rewards for batch on GPU
            batch_div_rew = get_batch_diversity_rewards(
                s=s,
                predictions=preds,
                len_states=s_len,
                padding_pos=padding_pos,
                topk_to_consider=topk_to_consider_div,
                embedding_layer=diversity_embedding,
                device=device,
                input_tokenizer=input_tokenizer,
                output_tokenizer=output_tokenizer,
            )
            total_diversity_rew += torch.sum(batch_div_rew)

            # Send preds and actions back to CPU
            preds = preds.to("cpu")
            a = a.to("cpu")

            # Calculate online novelty rew and add to cumulated nov_rew
            batch_nov_rew = get_batch_novelty_rewards(
                predictions=preds,
                unpopular_items=unpopular_actions_set,
                reward=novelty_rew_signal,
                topk_to_consider=topk_to_consider_nov,
            )
            total_novelty_rew += batch_nov_rew.sum()

            # Get number of hits/sum of ndcg in action-batch for each k
            hr_batch, ndcg_batch = get_hits_for_batch(
                predictions=preds, true_idx=a, top_k=topk_hr_ndcg
            )

            # Add k-hits/ndcg numpy array to current array to accumulate over batches
            hr += hr_batch
            ndcg += ndcg_batch

            # Add number of examples to total count
            n_total_samples += len(a)

            # Update actions-covered sets in topk-dict
            actions_covered_topk_dict = update_actions_covered_topk_dict(
                predictions=preds,
                actions_covered_topk_dict=actions_covered_topk_dict,
                top_k=topk_to_consider_cov,
                input_tokenizer=input_tokenizer,
                output_tokenizer=output_tokenizer,
            )

            # Calculate average number of repetitions for batch
            repetitions += get_batch_repetitions(
                s=s,
                preds=preds,
                topk=topk_hr_ndcg,
                input_tokenizer=input_tokenizer,
                output_tokenizer=output_tokenizer,
            )

        # Calculate ratios
        hr = hr / n_total_samples
        ndcg = ndcg / n_total_samples

        # Compute per sample reward averages
        avg_diversity_rew = total_diversity_rew / n_total_samples
        avg_novelty_rew = total_novelty_rew / n_total_samples

        # Calculate average loss over epoch
        loss = loss / len(evaluation_data_loader)

        # Calculate average repetitions per sample
        repetitions = repetitions / n_total_samples

        # Calculate coverage for epoch
        coverage_res = get_coverage(
            actions_covered_topk_dict=actions_covered_topk_dict,
            unpopular_set=unpopular_actions_set,
            num_actions=model.action_dim,
            topk=topk_to_consider_cov,
        )

    return loss, hr, ndcg, coverage_res, avg_diversity_rew, avg_novelty_rew, repetitions


def update_train_metrics(
    s,
    a,
    s_len,
    model,
    device,
    padding_pos,
    diversity_embedding,
    unpopular_actions_set,
    actions_covered_topk_dict,
    head_idx=0,
    topk_hr_ndcg=[5, 10, 20],
    topk_to_consider_div=1,
    topk_to_consider_nov=1,
    topk_to_consider_cov=[1, 5, 10],
    novelty_rew_signal=1,
    input_tokenizer=None,
    output_tokenizer=None,
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

        # Get predictions (on GPU)
        preds = get_preds(states=s, true_len=s_len, model=model, head_idx=head_idx)

        # Compute average diversity rewards for batch on GPU
        batch_div_rew = get_batch_diversity_rewards(
            s=s,
            predictions=preds,
            len_states=s_len,
            padding_pos=padding_pos,
            topk_to_consider=topk_to_consider_div,
            embedding_layer=diversity_embedding,
            device=device,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer,
        )

        batch_div_rew = torch.sum(batch_div_rew)

        # Send preds and actions back to CPU
        preds = preds.to("cpu")
        a = a.to("cpu")

        # Calculate online novelty rew and add to cumulated nov_rew
        batch_nov_rew = get_batch_novelty_rewards(
            predictions=preds,
            unpopular_items=unpopular_actions_set,
            reward=novelty_rew_signal,
            topk_to_consider=topk_to_consider_nov,
        )

        batch_nov_rew = batch_nov_rew.sum()

        # Get number of hits in action-batch for each k
        hr_batch, ndcg_batch = get_hits_for_batch(
            predictions=preds, true_idx=a, top_k=topk_hr_ndcg
        )

        # Update actions-covered sets in topk-dict
        actions_covered_topk_dict = update_actions_covered_topk_dict(
            predictions=preds,
            actions_covered_topk_dict=actions_covered_topk_dict,
            top_k=topk_to_consider_cov,
        )

        # Calculate average number of repetitions
        batch_repetitions = get_batch_repetitions(
            s=s,
            preds=preds,
            topk=topk_hr_ndcg,
            input_tokenizer=input_tokenizer,
            output_tokenizer=output_tokenizer,
        )

    return (
        hr_batch,
        ndcg_batch,
        actions_covered_topk_dict,
        batch_div_rew,
        batch_nov_rew,
        batch_repetitions,
    )
