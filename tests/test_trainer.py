import torch

from llm_rl.config import Config
from llm_rl.rollout import Group, Sample
from llm_rl.trainer import build_batch

PAD = 0


def make_sample(prompt_ids, response_ids, finish_reason="stop", reward=0.0):
    s = Sample(
        prompt="p",
        prompt_token_ids=list(prompt_ids),
        response_token_ids=list(response_ids),
        text="t",
        finish_reason=finish_reason,
    )
    s.reward_value = reward
    return s


def test_response_mask_aligns_with_logprob_indices():
    # prompt = [1,2,3], response = [4,5]; full = [1,2,3,4,5].
    # Logprob index t scores token t+1, so response tokens 4 and 5 (absolute 3 and 4)
    # are scored at shifted indices 2 and 3.
    group = Group(prompt="p", answer="1", samples=[make_sample([1, 2, 3], [4, 5])])
    batch = build_batch([group], torch.tensor([[1.0]]), PAD)
    assert batch.response_mask.tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert batch.input_ids.tolist() == [[1, 2, 3, 4, 5]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1]]


def test_advantage_is_broadcast_over_response_tokens_only():
    group = Group(prompt="p", answer="1", samples=[make_sample([1, 2], [3, 4, 5])])
    batch = build_batch([group], torch.tensor([[0.75]]), PAD)
    # Shifted indices 1,2,3 are the response; index 0 is the prompt interior.
    assert batch.advantages.tolist() == [[0.0, 0.75, 0.75, 0.75]]


def test_truncated_sequences_are_masked_out():
    group = Group(
        prompt="p",
        answer="1",
        samples=[
            make_sample([1, 2], [3, 4], finish_reason="stop"),
            make_sample([1, 2], [3, 4], finish_reason="length"),
        ],
    )
    batch = build_batch([group], torch.tensor([[1.0, 1.0]]), PAD, mask_truncated=True)
    assert batch.response_mask[0].sum() == 2.0
    assert batch.response_mask[1].sum() == 0.0
    # A masked sequence must also contribute no advantage.
    assert batch.advantages[1].abs().sum() == 0.0
    assert batch.metrics["batch/masked_out_frac"] == 0.5


def test_truncated_kept_when_masking_disabled():
    group = Group(prompt="p", answer="1", samples=[make_sample([1], [2, 3], "length")])
    batch = build_batch([group], torch.tensor([[1.0]]), PAD, mask_truncated=False)
    assert batch.response_mask.sum() == 2.0


def test_right_padding_and_ragged_lengths():
    group = Group(
        prompt="p",
        answer="1",
        samples=[
            make_sample([1, 2], [3]),
            make_sample([1, 2, 3], [4, 5, 6]),
        ],
    )
    batch = build_batch([group], torch.tensor([[1.0, -1.0]]), PAD)
    assert batch.input_ids.shape == (2, 6)
    # Padding goes on the right so the Gated DeltaNet recurrence never consumes a
    # pad token before real content.
    assert batch.input_ids[0].tolist() == [1, 2, 3, PAD, PAD, PAD]
    assert batch.attention_mask[0].tolist() == [1, 1, 1, 0, 0, 0]
    # Row 0: response token at absolute index 2 -> shifted index 1.
    assert batch.response_mask[0].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    # Row 1: response tokens at absolute 3,4,5 -> shifted 2,3,4.
    assert batch.response_mask[1].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_rewards_are_carried_through():
    group = Group(
        prompt="p",
        answer="1",
        samples=[make_sample([1], [2], reward=1.0), make_sample([1], [2], reward=0.0)],
    )
    batch = build_batch([group], torch.tensor([[0.5, -0.5]]), PAD)
    assert batch.rewards.tolist() == [1.0, 0.0]


def test_multiple_groups_index_into_advantages_correctly():
    groups = [
        Group(prompt="a", answer="1", samples=[make_sample([1], [2]), make_sample([1], [3])]),
        Group(prompt="b", answer="2", samples=[make_sample([4], [5]), make_sample([4], [6])]),
    ]
    adv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch = build_batch(groups, adv, PAD)
    # One response token each, at shifted index 0.
    assert [row[0].item() for row in batch.advantages] == [1.0, 2.0, 3.0, 4.0]


def test_config_yaml_roundtrip_and_defaults_inheritance():
    cfg = Config.from_yaml("configs/grpo.yaml")
    assert cfg.algo.name == "grpo"
    assert cfg.algo.clip_ratio_high == 0.28
    # Inherited from base.yaml.
    assert cfg.model.name == "Qwen/Qwen3.5-4B-Base"
    assert cfg.rollout.group_size == 8


def test_config_cli_overrides():
    cfg = Config.from_yaml("configs/grpo.yaml").apply_overrides(
        ["algo.clip_ratio_high=0.2", "rollout.group_size=16", "optim.lr=2e-6"]
    )
    assert cfg.algo.clip_ratio_high == 0.2
    assert cfg.rollout.group_size == 16
    assert cfg.optim.lr == 2e-6


def test_ppo_config_enables_value_head():
    cfg = Config.from_yaml("configs/ppo.yaml")
    assert cfg.model.value_head is True
    assert cfg.algo.advantage == "gae"


def test_algo_presets_validate():
    import pytest

    from llm_rl.algos import grpo, ppo, reinforce

    reinforce.validate(reinforce.preset())
    grpo.validate(grpo.preset())
    ppo.validate(ppo.preset(), Config.from_yaml("configs/ppo.yaml").model)

    bad = reinforce.preset()
    bad.ppo_epochs = 4
    with pytest.raises(ValueError):
        reinforce.validate(bad)


def test_pad_to_width_gives_constant_shapes():
    """Ragged batches fragment the allocator; fixed width is the fix that survives
    CUDA IPC (expandable_segments does not)."""
    short = Group(prompt="p", answer="1", samples=[make_sample([1, 2], [3])])
    long = Group(prompt="p", answer="1", samples=[make_sample(list(range(10)), list(range(10, 20)))])
    a = build_batch([short], torch.tensor([[1.0]]), PAD, pad_to=64)
    b = build_batch([long], torch.tensor([[1.0]]), PAD, pad_to=64)
    assert a.input_ids.shape == b.input_ids.shape == (1, 64)
    # Masks still mark only the real response tokens.
    assert a.response_mask.sum() == 1.0
    assert b.response_mask.sum() == 10.0


def test_pad_to_width_never_truncates():
    """A sequence longer than pad_to must keep its full length, not be cut."""
    g = Group(prompt="p", answer="1", samples=[make_sample(list(range(30)), list(range(30, 60)))])
    batch = build_batch([g], torch.tensor([[1.0]]), PAD, pad_to=16)
    assert batch.input_ids.shape == (1, 60)
    assert batch.response_mask.sum() == 30.0
