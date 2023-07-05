import torch
from recommenders.evaluate.diversity import get_last_action, get_batch_diversity_rewards


def test_get_last_action():
    state_front = torch.tensor([[0, 0, 1, 2, 3, 3], [0, 0, 0, 12, 13, 2]])
    state_back = torch.tensor([[1, 2, 3, 3, 0, 0], [12, 13, 2, 0, 0, 0]])
    len = torch.tensor([4, 3])

    # True last action
    true_last = torch.tensor([3, 2])

    end_res = get_last_action(state_back, padding_pos="end", s_len=len, device="cpu")
    front_res = get_last_action(
        s=state_front,
        padding_pos="beg",
    )

    assert torch.equal(end_res, true_last) & torch.equal(front_res, true_last)


def test_get_batch_diversity_rewards():
    preds = torch.tensor([[30, 20, 5, 10], [0, 10, 5, 2]])
    state_front = torch.tensor([[0, 0, 1, 2, 3, 0], [0, 0, 0, 12, 13, 1]])
    state_back = torch.tensor([[1, 2, 3, 0, 0, 0], [12, 13, 1, 0, 0, 0]])
    len = torch.tensor([4, 3])

    # Get embedding
    emb = torch.nn.Embedding.from_pretrained(
        torch.load("test/weightsemb.pt"), freeze=True
    )

    res = get_batch_diversity_rewards(
        state_back,
        preds,
        len,
        padding_pos="end",
        topk_to_consider=1,
        embedding_layer=emb,
        device="cpu",
    )

    res_front = get_batch_diversity_rewards(
        state_front,
        preds,
        len,
        padding_pos="beg",
        topk_to_consider=1,
        embedding_layer=emb,
        device="cpu",
    )

    assert torch.isclose(
        res,
        torch.tensor([0, 0]).float(),
    ).sum()

    assert torch.isclose(
        res_front,
        torch.tensor([0, 0]).float(),
    ).sum()
