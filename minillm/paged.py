# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Generated from 05_03_paged_kv_cache.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

import numpy as np

from minillm.config import LlamaConfig


class BlockPool:
    """Fixed-size blocks of KV cache memory, shared by all sequences."""

    def __init__(
        self, config: LlamaConfig, num_blocks: int, block_size: int = 16
    ) -> None:
        self.block_size = block_size
        shape = (
            num_blocks,
            block_size,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.keys = [
            np.zeros(shape, np.float32) for _ in range(config.num_hidden_layers)
        ]
        self.values = [
            np.zeros(shape, np.float32) for _ in range(config.num_hidden_layers)
        ]
        self.free = list(range(num_blocks))

    @property
    def num_blocks(self) -> int:
        return len(self.keys[0])

    def allocate(self) -> int:
        if not self.free:
            raise MemoryError("the KV cache pool is full")
        return self.free.pop()

    def available(self) -> int:
        """Returns how many blocks ``allocate`` can still hand out."""
        return len(self.free)

    def release(self, blocks: list[int]) -> None:
        self.free.extend(blocks)


class PagedCache:
    """One sequence's keys and values, stored in pool blocks."""

    def __init__(self, pool: BlockPool) -> None:
        self.pool = pool
        self.blocks: list[int] = []
        self.length = 0

    @property
    def num_layers(self) -> int:
        return len(self.pool.keys)

    def read(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        keys = self.pool.keys[layer][self.blocks].reshape(
            -1, *self.pool.keys[layer].shape[2:]
        )
        values = self.pool.values[layer][self.blocks].reshape(keys.shape)
        return keys[: self.length], values[: self.length]

    def append(self, layer: int, keys: np.ndarray, values: np.ndarray) -> None:
        size, start = self.pool.block_size, self.length
        if layer == 0:  # every layer stores the same positions
            while len(self.blocks) * size < start + len(keys):
                self.blocks.append(self.pool.allocate())
        for i in range(len(keys)):
            block, slot = self.blocks[(start + i) // size], (start + i) % size
            self.pool.keys[layer][block, slot] = keys[i]
            self.pool.values[layer][block, slot] = values[i]
        if layer == self.num_layers - 1:
            self.length += len(keys)

    def release(self) -> None:
        self.pool.release(self.blocks)
        self.blocks = []


from minillm.engine import Engine
from minillm.step import Request


class PagedEngine(Engine):
    """An engine whose requests share one pool of KV cache blocks."""

    def __init__(self, *args, pool: BlockPool, **kwargs) -> None:
        super().__init__(*args, new_cache=lambda _: PagedCache(pool), **kwargs)
        self.pool = pool
        self.preemptions = 0

    def step(self) -> None:
        # Admit waiting requests in arrival order while their tokens so far fit.
        spare, admitted = self.pool.available(), 0
        for request in self.waiting:
            if self.blocks_needed(request) > spare:
                break
            spare -= self.blocks_needed(request)
            admitted += 1
        held = [self.waiting.pop() for _ in range(len(self.waiting) - admitted)]
        prefill = (
            self.waiting
            and len(self.running) < self.max_batch
            and (self.continuous or not self.running)
        )
        if not prefill and not self.running and held:
            raise MemoryError(
                "a waiting request needs more blocks than the pool has"
            )
        if not prefill:
            held = self.preempt_until_decode_fits() + held
        super().step()
        self.waiting.extend(held)

    def preempt_until_decode_fits(self) -> list[Request]:
        """Frees the newest requests' blocks until every running request can
        store its next token, and returns them to recompute later."""
        preempted = []
        while self.blocks_to_grow() > self.pool.available():
            if len(self.running) == 1:
                raise MemoryError(
                    "a request needs more blocks than the pool has"
                )
            request, cache = self.running.pop()
            cache.release()
            preempted.insert(0, request)
            self.preemptions += 1
        return preempted

    def blocks_to_grow(self) -> int:
        size = self.pool.block_size
        return sum(c.length + 1 > len(c.blocks) * size for _, c in self.running)

    def blocks_needed(self, request: Request) -> int:
        # A preempted request recomputes the tokens it already generated.
        tokens = len(request.prompt) + len(request.output)
        return -(-tokens // self.pool.block_size)

    def prefill_tokens(self, request: Request, cache: PagedCache) -> list[int]:
        return request.prompt + request.output

    def release_finished(self, batch: list) -> None:
        for request, cache in batch:
            if request.done:
                cache.release()
