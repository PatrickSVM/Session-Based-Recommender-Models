import torch


def get_last_action(s, padding_pos, s_len=None, device="cuda"):
    """
    Extract last action from each state depending on padding.
    """
    if padding_pos == "end":
        idx = s_len.to(device) - 1
        return s.gather(index=idx.unsqueeze(1), dim=1).squeeze()
    else:
        return s[:, -1]


def get_batch_diversity_rewards(
    s,
    predictions,
    len_states,
    padding_pos,
    topk_to_consider,
    embedding_layer,
    device="cuda",
    input_tokenizer=None,
    output_tokenizer=None,
):
    """
    Comptues average cosine similarity between last action and top(k)prediciton.
    From this maps to reward. Cos: [-1, 1] -> rew [0, 2]

    GPU possible.

    s - (batch_size, state_len)
    predictions - (batch_size, action_space)
    len_states - (batch_size)
    pading_pos - 'beg'/'end'
    topk_to_consider - how many top predictions to consider
    embedding_layer - layer for embeddding actions

    cos_sim((batch_size, 1, embedding_dim), (batch_size, topk, embedding_dim)) -> (batch_size, topk)

    Returns batch diversity reward (batch_size).
    """
    # Init cosine similarity func
    cos_sim = torch.nn.CosineSimilarity(dim=2, eps=1e-6)

    # Get tensor (batch_size) of last actions
    last_actions = get_last_action(
        s=s, padding_pos=padding_pos, s_len=len_states, device=device
    )

    # Choose topk preds to consider
    _, action_idx = torch.topk(predictions, k=topk_to_consider)

    # Translate to input tokens
    if input_tokenizer != None:
        action_idx = action_idx.to("cpu")
        action_idx = action_idx.apply_(
            lambda x: input_tokenizer.stoi(output_tokenizer.itos(x))
        )
        action_idx = action_idx.to(device)

    # Embed last action and predicted actions
    embed_last_a = embedding_layer(last_actions).unsqueeze(1)
    embed_preds = embedding_layer(action_idx)

    # Get similarity
    similarity = cos_sim(embed_last_a, embed_preds)

    # Take mean over rows to get avg sim of top preds to last a
    mean_sim = torch.mean(similarity, dim=1)
    batch_rewards = 1 - mean_sim

    return batch_rewards
