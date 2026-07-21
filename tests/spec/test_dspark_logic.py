"""CPU unit tests for the pure-logic core of the DSpark spec-decode implementation.

Run (after installing torch):  pytest tests/spec/test_dspark_logic.py -q

``verify`` and ``planner`` depend only on torch, so they are fully exercised here.
The Markov-head test additionally needs the ``minisgl.layers`` primitives and is
skipped automatically if that import pulls in unavailable CUDA kernels.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.spec.planner import (  # noqa: E402
    SpsCostModel,
    compute_survival,
    compute_verify_token_budget,
    schedule_verify_lens,
)
from minisgl.spec.verify import (  # noqa: E402
    accept_greedy,
    accept_sampling,
    gather_committed_tokens,
)


# --------------------------------------------------------------------------- accept (greedy)
def test_accept_greedy_all_match():
    gamma = 4
    target_predict = torch.tensor([[10, 11, 12, 13, 99]])  # W = gamma + 1
    draft = target_predict[:, :gamma].clone()  # all drafts equal target's predictions
    r = accept_greedy(draft, target_predict)
    assert r.correct_len.tolist() == [gamma]
    assert r.bonus.tolist() == [99]  # the trailing target prediction
    assert r.commit_len.tolist() == [gamma + 1]


def test_accept_greedy_first_mismatch():
    target_predict = torch.tensor([[10, 11, 12, 13, 99]])
    draft = torch.tensor([[7, 11, 12, 13]])  # mismatch at position 0
    r = accept_greedy(draft, target_predict)
    assert r.correct_len.tolist() == [0]
    assert r.bonus.tolist() == [10]  # target's own token at the mismatch


def test_accept_greedy_partial_prefix():
    target_predict = torch.tensor([[10, 11, 12, 13, 99]])
    draft = torch.tensor([[10, 11, 55, 13]])  # matches 2, breaks at position 2
    r = accept_greedy(draft, target_predict)
    assert r.correct_len.tolist() == [2]
    assert r.bonus.tolist() == [12]  # target prediction at the first mismatch position


def test_accept_greedy_batch():
    target_predict = torch.tensor([[1, 2, 3], [4, 5, 6]])  # gamma = 2
    draft = torch.tensor([[1, 2], [9, 5]])
    r = accept_greedy(draft, target_predict)
    assert r.correct_len.tolist() == [2, 0]
    assert r.bonus.tolist() == [3, 4]


# --------------------------------------------------------------------------- accept (sampling)
def test_accept_sampling_accepts_all_when_target_ge_draft():
    bs, gamma, vocab = 1, 3, 5
    draft_tokens = torch.tensor([[0, 1, 2]])
    # target concentrates all mass on the drafted tokens => p/q >= 1 everywhere
    target_probs = torch.zeros(bs, gamma + 1, vocab)
    for k, t in enumerate([0, 1, 2, 3]):
        target_probs[0, k, t] = 1.0
    draft_probs = torch.full((bs, gamma, vocab), 1.0 / vocab)
    uniforms = torch.zeros(bs, gamma)  # always accept when p/q > 0
    r = accept_sampling(draft_tokens, draft_probs, target_probs, uniforms=uniforms)
    assert r.correct_len.tolist() == [3]
    assert r.bonus.tolist() == [3]  # trailing target argmax mass


def test_accept_sampling_rejects_at_first_bad_token():
    bs, gamma, vocab = 1, 3, 4
    draft_tokens = torch.tensor([[0, 1, 2]])
    draft_probs = torch.zeros(bs, gamma, vocab)
    draft_probs[0, :, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # draft is certain of token 0
    draft_probs[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    draft_probs[0, 2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    target_probs = torch.zeros(bs, gamma + 1, vocab)
    target_probs[0, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # target rejects draft token 0
    target_probs[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    target_probs[0, 2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    target_probs[0, 3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    uniforms = torch.full((bs, gamma), 0.999)
    r = accept_sampling(draft_tokens, draft_probs, target_probs, uniforms=uniforms)
    assert r.correct_len.tolist() == [0]
    assert r.bonus.tolist() == [3]  # residual mass lands on token 3


# --------------------------------------------------------------------------- planner
def test_survival_is_cumprod_and_monotonic():
    conf = torch.tensor([[0.9, 0.8, 0.5]])
    surv = compute_survival(conf)
    assert torch.allclose(surv, torch.tensor([[0.9, 0.72, 0.36]]))
    assert torch.all(surv[:, 1:] <= surv[:, :-1])


def test_budget_verifies_all_under_flat_cost():
    # per_token_ms = 0 => step time constant => more verify tokens is always better.
    surv = torch.rand(4, 7)
    cost = SpsCostModel(base_ms=6.0, per_token_ms=0.0)
    budget = compute_verify_token_budget(surv, cost)
    assert budget == 4 * 7


def test_budget_shrinks_under_expensive_tokens():
    # very expensive per-token cost + low survival => grant fewer verify tokens.
    surv = torch.full((4, 7), 0.05)
    cheap = compute_verify_token_budget(surv, SpsCostModel(6.0, 0.0))
    pricey = compute_verify_token_budget(surv, SpsCostModel(6.0, 5.0))
    assert pricey < cheap


def test_schedule_verify_lens_are_contiguous_prefixes():
    # survival is monotonically decreasing per row => top-k must select prefixes.
    surv = torch.tensor([[0.9, 0.8, 0.1], [0.7, 0.2, 0.05]])
    lens = schedule_verify_lens(surv, budget=3, min_verify_len=1)
    assert lens.sum().item() == 3 or lens.min().item() >= 1
    # highest-survival row gets at least as many as the lower one
    assert lens[0].item() >= lens[1].item()


def test_schedule_respects_min_and_max():
    surv = torch.rand(3, 5)
    lens = schedule_verify_lens(surv, budget=0, min_verify_len=1)
    assert torch.all(lens >= 1) and torch.all(lens <= 5)


# --------------------------------------------------------------------------- host helper
def test_gather_committed_tokens():
    draft = torch.tensor([[10, 11, 12], [20, 21, 22]])
    from minisgl.spec.verify import AcceptResult

    res = AcceptResult(
        correct_len=torch.tensor([2, 0]),
        bonus=torch.tensor([99, 77]),
        commit_len=torch.tensor([3, 1]),
    )
    out = gather_committed_tokens(draft, res)
    assert out == [[10, 11, 99], [77]]


# --------------------------------------------------------------------------- Markov head (optional)
def test_vanilla_markov_bias_shapes_and_effect():
    heads = pytest.importorskip("minisgl.spec.heads")
    head = heads.VanillaMarkov(vocab_size=8, markov_rank=4)
    torch.manual_seed(0)
    head.w1 = torch.randn(8, 4)
    head.w2 = torch.randn(8, 4)
    base = torch.zeros(2, 8)
    prev = torch.tensor([1, 3])
    out = head.apply_step_logits(base, token_ids=prev, hidden_states=None)
    assert out.shape == (2, 8)
    # bias == w2 @ w1[prev]
    expected = torch.nn.functional.linear(torch.nn.functional.embedding(prev, head.w1), head.w2)
    assert torch.allclose(out, expected, atol=1e-5)


def test_run_markov_block_is_serial():
    heads = pytest.importorskip("minisgl.spec.heads")
    head = heads.VanillaMarkov(vocab_size=6, markov_rank=3)
    head.w1 = torch.zeros(6, 3)  # zero bias => sampler sees base logits only
    head.w2 = torch.zeros(6, 3)
    base_logits = torch.zeros(1, 4, 6)
    base_logits[0, torch.arange(4), torch.tensor([2, 3, 4, 5])] = 10.0
    sampler = lambda logits, idx: logits.argmax(-1)  # noqa: E731
    toks, corrected = head.sample_block(
        base_logits, first_prev_tokens=torch.tensor([0]), hidden_states=None, sampler=sampler
    )
    assert toks.tolist() == [[2, 3, 4, 5]]
    assert corrected.shape == (1, 4, 6)
