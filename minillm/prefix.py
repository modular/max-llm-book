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
"""Generated from 05_04_prefix_caching.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

from collections import OrderedDict

from minillm.config import LlamaConfig
from minillm.paged import BlockPool, PagedCache, PagedEngine
from minillm.step import Request


class PrefixCachingPool(BlockPool):
    """A block pool that remembers full prompt blocks for reuse."""

    def __init__(
        self, config: LlamaConfig, num_blocks: int, block_size: int = 16
    ) -> None:
        super().__init__(config, num_blocks, block_size)
        self.refs = [0] * num_blocks
        self.by_hash: dict[int, int] = {}
        self.hash_of: dict[int, int] = {}
        self.unused: OrderedDict[int, None] = OrderedDict()  # cached, no users

    def allocate(self) -> int:
        if self.free:
            block = self.free.pop()
        elif self.unused:
            block, _ = self.unused.popitem(last=False)  # evict the oldest
            del self.by_hash[self.hash_of.pop(block)]
        else:
            raise MemoryError("the KV cache pool is full")
        self.refs[block] = 1
        return block

    def available(self) -> int:
        # Cached blocks that no request uses can be evicted and reused.
        return len(self.free) + len(self.unused)

    def release(self, blocks: list[int]) -> None:
        for block in blocks:
            self.refs[block] -= 1
            if self.refs[block] == 0:
                if block in self.hash_of:
                    self.unused[block] = None
                else:
                    self.free.append(block)

    def block_hashes(self, tokens: list[int]) -> list[int]:
        hashes, previous = [], 0
        for start in range(
            0, len(tokens) - self.block_size + 1, self.block_size
        ):
            previous = hash(
                (previous, tuple(tokens[start : start + self.block_size]))
            )
            hashes.append(previous)
        return hashes

    def lookup(self, prompt: list[int]) -> list[int]:
        """Returns the cached blocks that start ``prompt``, and claims them.
        The last prompt token is always left to compute, since its logits
        are needed."""
        blocks = []
        for h in self.block_hashes(prompt[:-1]):
            block = self.by_hash.get(h)
            if block is None:
                break
            self.refs[block] += 1
            self.unused.pop(block, None)
            blocks.append(block)
        return blocks

    def register(self, prompt: list[int], blocks: list[int]) -> None:
        """Records a sequence's full prompt blocks for later requests."""
        # The last block may be partial; only full blocks have hashes.
        for h, block in zip(self.block_hashes(prompt), blocks, strict=False):
            if h not in self.by_hash:
                self.by_hash[h] = block
                self.hash_of[block] = h


class PrefixCachingEngine(PagedEngine):
    """A paged engine that skips computing prompt prefixes it has cached."""

    def __init__(self, *args, pool: PrefixCachingPool, **kwargs) -> None:
        super().__init__(*args, pool=pool, **kwargs)
        self.computed_prompt_tokens = 0

    def prefill_tokens(self, request: Request, cache: PagedCache) -> list[int]:
        tokens = (
            request.prompt + request.output
        )  # output is empty unless preempted
        cache.blocks = self.pool.lookup(tokens)
        cache.length = len(cache.blocks) * self.pool.block_size
        new = tokens[cache.length :]
        self.computed_prompt_tokens += len(new)
        return new

    def prefilled(self, request: Request, cache: PagedCache) -> None:
        self.pool.register(request.prompt, cache.blocks)
