import math

import pytest
import torch

from eyetrack2llm.torch import (
    DirectedBilinearHead,
    masked_jsd,
    masked_softmax,
    word_pool,
)


def test_word_pool_mean_and_gradient():
    states = torch.tensor(
        [[[1.0, 3.0], [3.0, 5.0], [7.0, 9.0], [100.0, 100.0]]],
        requires_grad=True,
    )
    pooled, mask = word_pool(states, torch.tensor([[0, 0, 1, -1]]), n_words=3)
    torch.testing.assert_close(
        pooled, torch.tensor([[[2.0, 4.0], [7.0, 9.0], [0.0, 0.0]]])
    )
    assert mask.tolist() == [[True, True, False]]
    pooled.sum().backward()
    assert torch.isfinite(states.grad).all()


def test_directed_head_is_asymmetric():
    head = DirectedBilinearHead(2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]))
    states = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    logits, mask = head(states, torch.tensor([[True, True]]))
    assert logits[0, 0, 0, 1] != logits[0, 0, 1, 0]
    assert mask.shape == (1, 1, 2, 2)


def test_masked_softmax_all_masked_is_zero():
    output = masked_softmax(torch.tensor([[1.0, 2.0]]), torch.tensor([[False, False]]))
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_masked_jsd_known_values_and_gradient():
    identical = masked_jsd(
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[True, True]]),
        input_is_logits=False,
    )
    assert identical.item() == pytest.approx(0.0)

    disjoint = masked_jsd(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[True, True]]),
        input_is_logits=False,
    )
    assert disjoint.item() == pytest.approx(math.log(2), rel=1e-6)

    logits = torch.tensor([[1.0, -1.0], [2.0, 2.0]], requires_grad=True)
    target = torch.tensor([[0.25, 0.75], [0.0, 0.0]])
    loss = masked_jsd(logits, target, torch.ones_like(target, dtype=torch.bool))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
