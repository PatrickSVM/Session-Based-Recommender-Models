import random
import numpy as np
import torch


def simulate_trajectories(model, top_preds_to_show, num_traj, start="rand"):
    """
    Simulated customer picks one of the topk preds. Function yields
    num_traj sampled trajectories.

    top_preds_to_show: From how many can siumulated customer choose?

    TODO: This assumes that action_dim=item_num - ID of action is the same ID as item!
    """
    if start == "rand":
        start_item = random.randint(0, model.item_num)
    else:
        start_item = start


def get_batch_repetitions(
    s,
    preds,
    topk=[1],
    input_tokenizer=None,
    output_tokenizer=None,
):
    """
    Takes in states and predicitons and returns summed repetitions
    per topk-prediction vector, so over samples in the batch.

    Actions are tokenized by output_tokenizer which first needs
    to be reversed to check repetitions.
    """
    s = s.cpu().numpy()
    s = np.expand_dims(s, axis=-1)

    # Choose topk preds to consider
    _, action_idx = torch.topk(preds, k=max(topk))

    # Translate to input tokens
    if input_tokenizer != None:
        action_idx = action_idx.to("cpu")
        action_idx = action_idx.apply_(
            lambda x: input_tokenizer.stoi(output_tokenizer.itos(x))
        )
    action_idx = action_idx.numpy()

    rep_res = np.zeros(len(topk))
    for i, k in enumerate(topk):
        # Get sum of repetitions per row
        repetitions = (s == np.expand_dims(action_idx[:, :k], axis=1)).sum(-1)

        # Save summed repetitions for all rows
        rep_res[i] = repetitions.sum(1).sum()

    return rep_res
