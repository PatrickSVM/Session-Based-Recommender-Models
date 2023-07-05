import torch
import numpy as np
from recommenders.evaluate.repetitiveness import get_batch_repetitions


def test_get_batch_repetitions():
    s = torch.tensor([[1, 1, 2, 2, 3, 4], [1, 2, 3, 4, 5, 6], [1, 1, 2, 2, 3, 4]])
    preds = torch.tensor([[0, 11, 10, 5, 5], [0, 11, 10, 9, 8], [9, 8, 7, 10, -10]])

    # real:
    # k=1: (2+1+1)/3=1.3333
    # k=2: (4+2+1)/3=2.3333
    # k=5: (6+4+6)/3=5.3333
    result = get_batch_repetitions(s=s, preds=preds, topk=[1, 2, 5])

    assert np.isclose(result[0] / 3, 1.33333333)
    assert np.isclose(result[1] / 3, 2.33333333)
    assert np.isclose(result[2] / 3, 5.33333333)
