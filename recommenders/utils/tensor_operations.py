import torch


def gather_from_3d(q_tensor, action_idxs, num_heads=3):
    """
    Function that takes a 3D tensor of Q-values and gathers
    the value of the max action for each batch and each
    of the Q-heads.

    Input:
    q_tensor: (batch_size, q_heads, action_dim)
    action_idxs: (batch_size)

    For each sample, we want to get the values of all heads
    referring to the one max chosen action for this sample.
    So we choose the same action for each head per sample.
    action_idxs contains the indexes of these actions.

    Return:
    gathered_q_vals: (batch_size, q_heads)
    """
    # First unsqueeze it to (batch_size, 1)
    action_idxs = action_idxs.unsqueeze(1)

    # Then repeat it num_heads times on second dimension and
    # unsqueeze again to (batch_size, num_heads, 1)
    action_idxs = action_idxs.repeat(1, num_heads).unsqueeze(2)
    gathered_q_vals = torch.gather(q_tensor, dim=2, index=action_idxs)

    # Squeeze result
    gathered_q_vals = gathered_q_vals.squeeze()

    return gathered_q_vals


def weighted_q_loss(q_vals, w):
    """
    Compute matrix multiplication between q_loss of
    the correspondin sample actions and the weight vector
    w.

    w: (q_heads)
    q_vals: (batch_size, q_heads)

    (batch_size, q_heads) x (q_heads) -> (batch_size)
    """
    return torch.matmul(q_vals, w)


def get_weighted_q_target(q_vals, w):
    """
    Compute batched matrix multiplication of 3d tensor
    and provided weights.

    w: (q_heads)
    q_vals: (batch_size, q_heads, action_dim)

    w x (batch_size, q_heads, action_dim)  -> (batch_size, action_dim)

    Return:
    q_vals (batch_size, action_dim)
    """
    return torch.matmul(w, q_vals)


def get_max_action(q_vals):
    """
    Take tensor of q_vals and return index of max
    value per sample in batch.

    q_vals: (batch_size, action_dim)

    Return:
    max_action_idxs (batch_size)
    """

    return torch.argmax(q_vals, dim=1)
