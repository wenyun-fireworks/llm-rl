"""Rollout generation against a vLLM server.

RL on LLMs is generation-bound: a single step here samples group_size completions
for every prompt, which dwarfs the cost of the backward pass. We therefore drive a
real vLLM server (continuous batching, paged KV cache, prompt-prefix sharing across
the group) rather than HF `generate`.

Token ids come back from the server directly via `return_token_ids`. That matters:
re-tokenizing the returned text is not guaranteed to round-trip, and any mismatch
between the tokens that were sampled and the tokens we score would silently corrupt
the importance ratios that PPO and GRPO depend on.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field

import requests

from .config import RolloutConfig


@dataclass
class Sample:
    """One completion for one prompt."""

    prompt: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    text: str
    finish_reason: str

    @property
    def truncated(self) -> bool:
        """True when the sampler hit max_tokens instead of emitting a stop token.

        A truncated rollout has no final answer, so its reward is uninformative and
        DAPO masks it out of the loss.
        """
        return self.finish_reason == "length"

    @property
    def total_length(self) -> int:
        return len(self.prompt_token_ids) + len(self.response_token_ids)


@dataclass
class Group:
    """The `group_size` samples drawn for a single prompt, plus its gold answer."""

    prompt: str
    answer: str
    samples: list[Sample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)


class VLLMServer:
    """Launches and owns a `vllm serve` subprocess configured for IPC weight sync."""

    def __init__(
        self,
        model: str,
        port: int = 8000,
        gpu_memory_utilization: float = 0.35,
        max_model_len: int = 4096,
        device: int | None = None,
        enforce_eager: bool = False,
        extra_args: list[str] | None = None,
    ):
        self.model = model
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.device = device
        self.enforce_eager = enforce_eager
        self.extra_args = extra_args or []
        self.process: subprocess.Popen | None = None

    def start(self, timeout: float = 900.0) -> VLLMServer:
        args = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            # The trainer pushes weights straight into the engine's GPU memory over
            # CUDA IPC, which is zero-copy when the two are colocated on one device.
            "--weight-transfer-config",
            '{"backend": "ipc"}',
        ]
        if self.enforce_eager:
            args.append("--enforce-eager")
        args += self.extra_args

        env = dict(os.environ)
        # IPC handles ride the HTTP control plane as pickled blobs, which vLLM
        # refuses to deserialize unless this is set on both ends.
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        if self.device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.device)

        self.process = subprocess.Popen(args, env=env)
        self.wait_until_ready(timeout)
        return self

    def wait_until_ready(self, timeout: float = 900.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"vllm serve exited with code {self.process.returncode}")
            try:
                if requests.get(f"{self.url}/health", timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2.0)
        raise TimeoutError(f"vllm server did not become ready within {timeout}s")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.process = None

    def __enter__(self) -> VLLMServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class Rollout:
    """Samples groups of completions from a running vLLM server."""

    def __init__(self, url: str, model: str, cfg: RolloutConfig, timeout: float = 3600.0):
        self.url = url.rstrip("/")
        self.model = model
        self.cfg = cfg
        self.timeout = timeout

    def generate(
        self,
        prompts: list[str],
        answers: list[str],
        group_size: int | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> list[Group]:
        """Draw `group_size` samples for each prompt.

        All prompts go out in a single request so vLLM can batch them continuously
        and share each prompt's KV cache across its group.
        """
        if len(prompts) != len(answers):
            raise ValueError(f"got {len(prompts)} prompts but {len(answers)} answers")
        n = group_size or self.cfg.group_size

        payload = {
            "model": self.model,
            "prompt": prompts,
            "n": n,
            "max_tokens": max_new_tokens or self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p if top_p is None else top_p,
            "return_token_ids": True,
        }
        if self.cfg.top_k and self.cfg.top_k > 0:
            payload["top_k"] = self.cfg.top_k
        request_seed = self.cfg.seed if seed is None else seed
        if request_seed is not None:
            payload["seed"] = request_seed

        body = self._post_with_retry(payload)
        return self._to_groups(body, prompts, answers, n)

    def _post_with_retry(self, payload: dict, attempts: int = 4) -> dict:
        """POST with backoff.

        Transient failures (a busy server, a dropped connection) should not kill a
        multi-hour training run. A dead engine will still surface after the retries
        are exhausted, which is the behaviour we want: unrecoverable, so stop.
        """
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = requests.post(
                    f"{self.url}/v1/completions", json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as error:
                last_error = error
                if attempt < attempts - 1:
                    time.sleep(min(2**attempt * 5, 60))
        raise RuntimeError(f"vLLM generate failed after {attempts} attempts: {last_error}")

    def _to_groups(
        self, body: dict, prompts: list[str], answers: list[str], n: int
    ) -> list[Group]:
        groups = [Group(prompt=p, answer=a) for p, a in zip(prompts, answers)]
        for choice in body["choices"]:
            # With a list of prompts and n samples each, choices arrive flattened as
            # prompt-major: indices [i*n, (i+1)*n) belong to prompt i.
            prompt_index = choice["index"] // n
            token_ids = choice.get("token_ids")
            prompt_token_ids = choice.get("prompt_token_ids")
            if token_ids is None or prompt_token_ids is None:
                raise RuntimeError(
                    "vLLM did not return token ids; the server must support "
                    "`return_token_ids` or the sampled tokens cannot be scored exactly"
                )
            groups[prompt_index].samples.append(
                Sample(
                    prompt=prompts[prompt_index],
                    prompt_token_ids=list(prompt_token_ids),
                    response_token_ids=list(token_ids),
                    text=choice["text"],
                    finish_reason=choice.get("finish_reason") or "unknown",
                )
            )
        missing = [i for i, g in enumerate(groups) if len(g) != n]
        if missing:
            raise RuntimeError(f"expected {n} samples per prompt; prompts {missing} came up short")
        return groups


def rollout_stats(groups: list[Group]) -> dict[str, float]:
    samples = [s for g in groups for s in g.samples]
    if not samples:
        return {}
    lengths = [len(s.response_token_ids) for s in samples]
    return {
        "rollout/num_samples": float(len(samples)),
        "rollout/response_len_mean": sum(lengths) / len(lengths),
        "rollout/response_len_max": float(max(lengths)),
        "rollout/truncated_frac": sum(s.truncated for s in samples) / len(samples),
    }
