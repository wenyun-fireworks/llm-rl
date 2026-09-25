"""Push updated policy weights from the trainer into the vLLM engine.

Reloading a checkpoint between RL steps would dominate the step time. Instead we use
vLLM's first-class weight transfer with the CUDA IPC backend: when the trainer and
the engine sit on the same physical GPU, the transfer is zero-copy — the engine maps
the trainer's parameter storage directly.

The control plane (handshake, start/update/finish) rides HTTP; only the pickled IPC
handles cross the wire, never the weights themselves.
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo
from vllm.distributed.weight_transfer.packed_tensor import DEFAULT_PACKED_BUFFER_SIZE_BYTES

_MIB = 1024 * 1024


def _nbytes(meta) -> int:
    count = 1
    for dim in meta.shape:
        count *= dim
    return count * torch.empty(0, dtype=meta.dtype).element_size()


def _round_up_mib(nbytes: int) -> int:
    return ((nbytes + _MIB - 1) // _MIB) * _MIB


class TextOnlyModuleSource(ModuleSource):
    """A ModuleSource that skips frozen parameters and casts to the engine's dtype.

    Two jobs:

    * Skip the vision tower. It is frozen, so re-shipping it every step is pure
      overhead; the engine already loaded it from the checkpoint at startup.
    * Cast fp32 master weights down to the engine's bf16. The trainer keeps fp32
      parameters so that small Adam updates are not rounded away, but vLLM holds a
      bf16 model and the transfer must match its dtype.

    metadata() and iteration must agree element for element, so the cast is applied
    to both channels.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        skip_prefixes: tuple[str, ...] = (),
        cast_dtype: torch.dtype | None = torch.bfloat16,
    ):
        super().__init__(module)
        self._skip_prefixes = skip_prefixes
        self._cast_dtype = cast_dtype

    def _keep(self, name: str) -> bool:
        return not name.startswith(self._skip_prefixes)

    def metadata(self):
        metas = [meta for meta in super().metadata() if self._keep(meta.name)]
        if self._cast_dtype is None:
            return metas
        return [replace(meta, dtype=self._cast_dtype) for meta in metas]

    def __iter__(self):
        for name, tensor in super().__iter__():
            if not self._keep(name):
                continue
            yield name, (tensor.to(self._cast_dtype) if self._cast_dtype else tensor)


class WeightSync:
    """Trainer-side handle that pushes weights to a vLLM server once per RL step."""

    def __init__(
        self,
        model: torch.nn.Module,
        server_url: str,
        rank: int = 0,
        packed: bool = True,
        skip_prefixes: tuple[str, ...] = ("model.visual.",),
        cast_dtype: torch.dtype | None = torch.bfloat16,
    ):
        # Must match the server's environment or it will refuse the pickled handles.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        self.source = TextOnlyModuleSource(
            model, skip_prefixes=skip_prefixes, cast_dtype=cast_dtype
        )
        metadata = self.source.metadata()
        self.num_synced = len(metadata)

        # The packed producer streams parameters through one reusable buffer, so the
        # buffer must hold the single largest tensor. Qwen3.5's tied embedding over a
        # 248,320-token vocab is 1.27 GB at 4B, which overflows vLLM's 1 GiB default.
        largest = max(_nbytes(meta) for meta in metadata) if metadata else 0
        buffer_bytes = max(DEFAULT_PACKED_BUFFER_SIZE_BYTES, _round_up_mib(largest))

        self.engine = WeightTransferTrainerFactory.trainer_init(
            # packed=True streams through one bounded buffer instead of exporting an
            # IPC handle per tensor (there are several hundred).
            init_info=IPCTrainerInitInfo(
                rank=rank, packed=packed, packed_buffer_size_bytes=buffer_bytes
            ),
            client=HTTPVLLMWeightSyncClient(server_url),
            source=self.source,
        )
        self.buffer_bytes = buffer_bytes

    def sync(self) -> None:
        """Ship the current weights. Called on every trainer rank."""
        self.engine.send_weights()

    def shutdown(self) -> None:
        self.engine.shutdown()
