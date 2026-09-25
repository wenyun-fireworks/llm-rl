"""Integration checks against a live vLLM server.

Skipped unless a server is reachable. Point them at one with:

    VLLM_URL=http://127.0.0.1:8100 pytest tests/test_gpu_integration.py -v

The trainer must run on the same physical GPU as the server for the weight-sync
test, because CUDA IPC resolves handles by GPU UUID.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.gpu

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8100")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-4B-Base")


def server_up() -> bool:
    try:
        import requests

        return requests.get(f"{URL}/health", timeout=3).status_code == 200
    except Exception:
        return False


# Opt-in only. Detecting "some server on port 8100" is not enough: it may be serving
# a different model than VLLM_MODEL, in which case these fail confusingly rather
# than skipping. Run with RUN_GPU_TESTS=1 VLLM_URL=... VLLM_MODEL=...
ENABLED = os.environ.get("RUN_GPU_TESTS") == "1" and server_up()
needs_server = pytest.mark.skipif(
    not ENABLED, reason="set RUN_GPU_TESTS=1 with a matching vLLM server to run"
)


@pytest.fixture(scope="module")
def rollout():
    from llm_rl.config import RolloutConfig
    from llm_rl.rollout import Rollout

    return Rollout(URL, MODEL, RolloutConfig(group_size=2, max_new_tokens=64, seed=0))


@needs_server
def test_rollout_returns_exact_token_ids(rollout):
    """Re-tokenizing generated text is not guaranteed to round-trip, so the
    sampled ids must come back from the server directly."""
    from llm_rl.data import Example

    example = Example(problem="What is 6 times 7?", answer="42")
    groups = rollout.generate([example.prompt], [example.answer])

    assert len(groups) == 1 and len(groups[0].samples) == 2
    for sample in groups[0].samples:
        assert sample.prompt_token_ids and sample.response_token_ids
        assert sample.finish_reason in {"stop", "length"}
        assert sample.truncated == (sample.finish_reason == "length")


@needs_server
def test_prompt_tokenization_matches_huggingface(rollout):
    from transformers import AutoTokenizer

    from llm_rl.data import Example

    example = Example(problem="What is 6 times 7?", answer="42")
    groups = rollout.generate([example.prompt], [example.answer])
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf_ids = tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
    assert groups[0].samples[0].prompt_token_ids == hf_ids


@needs_server
@pytest.mark.slow
def test_logprob_parity_between_vllm_and_hf():
    """vLLM and HF must agree closely, or every importance ratio starts wrong.

    They will not agree exactly: different kernels and bf16 rounding leave a
    residual, which is precisely why the trainer recomputes old logprobs rather
    than reusing the sampler's.
    """
    import requests

    from llm_rl.config import ModelConfig
    from llm_rl.data import Example
    from llm_rl.logprobs import compute_logprobs
    from llm_rl.model import load_policy

    prompt = Example(problem="What is 17 times 24?", answer="408").prompt
    body = requests.post(
        f"{URL}/v1/completions",
        json={
            "model": MODEL, "prompt": prompt, "n": 1, "max_tokens": 32,
            "temperature": 1.0, "seed": 7, "logprobs": 0, "return_token_ids": True,
        },
        timeout=600,
    ).json()["choices"][0]

    vllm_logprobs = torch.tensor(body["logprobs"]["token_logprobs"], dtype=torch.float32)
    ids = body["prompt_token_ids"] + body["token_ids"]

    policy = load_policy(ModelConfig(gradient_checkpointing=False), device="cuda")
    input_ids = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        out = compute_logprobs(policy, input_ids, torch.ones_like(input_ids))

    start = len(body["prompt_token_ids"]) - 1
    hf_logprobs = out.logprobs[0, start : start + len(body["token_ids"])].float().cpu()

    correlation = torch.corrcoef(torch.stack([hf_logprobs, vllm_logprobs]))[0, 1]
    assert correlation > 0.99, f"logprob correlation only {correlation:.4f}"
    assert (hf_logprobs - vllm_logprobs).abs().mean() < 0.05


@needs_server
@pytest.mark.slow
def test_weight_sync_is_faithful_and_reversible():
    """Syncing unchanged weights must not alter output; syncing changed weights
    must. Requires the trainer to be on the server's GPU."""
    import requests

    from llm_rl.config import ModelConfig
    from llm_rl.data import Example
    from llm_rl.model import load_policy
    from llm_rl.weight_sync import WeightSync

    prompt = Example(problem="What is 17 times 24?", answer="408").prompt

    def greedy():
        return requests.post(
            f"{URL}/v1/completions",
            json={"model": MODEL, "prompt": prompt, "n": 1, "max_tokens": 24,
                  "temperature": 0.0, "seed": 0},
            timeout=600,
        ).json()["choices"][0]["text"]

    baseline = greedy()
    policy = load_policy(ModelConfig(gradient_checkpointing=False), device="cuda")
    sync = WeightSync(policy.model, URL)

    sync.sync()
    assert greedy() == baseline, "a no-op sync changed the engine's output"

    target = policy.backbone.layers[0].mlp.gate_proj.weight
    original = target.detach().clone()
    with torch.no_grad():
        target.add_(torch.randn_like(target) * 0.05)
    sync.sync()
    assert greedy() != baseline, "a perturbed sync did not reach the engine"

    # Always leave the engine on clean weights, or every later test and eval in
    # this session silently measures a broken model.
    with torch.no_grad():
        target.copy_(original)
    sync.sync()
    assert greedy() == baseline, "failed to restore the engine's weights"
