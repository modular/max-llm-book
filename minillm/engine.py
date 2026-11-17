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
"""Generated from 05_02_continuous_batching.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from max.driver import Device

from minillm.config import LlamaConfig
from minillm.step import ContiguousCache, Request, Step, run_step


class Engine:
    """Serves requests with a batching scheduler and greedy decoding."""

    def __init__(
        self,
        step: Step,
        config: LlamaConfig,
        device: Device,
        max_batch: int = 4,
        continuous: bool = True,
        new_cache: Callable[[LlamaConfig], object] = ContiguousCache,
    ) -> None:
        self.step_fn = step
        self.config = config
        self.device = device
        self.max_batch = max_batch
        self.continuous = continuous
        self.new_cache = new_cache
        self.waiting: deque[Request] = deque()
        self.running: list[tuple[Request, object]] = []
        self.steps = 0

    def add(self, request: Request) -> None:
        self.waiting.append(request)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def step(self) -> None:
        room = self.max_batch - len(self.running)
        can_admit = (
            self.waiting and room > 0 and (self.continuous or not self.running)
        )
        if can_admit:
            admitted = [
                self.waiting.popleft()
                for _ in range(min(room, len(self.waiting)))
            ]
            batch = [(r, self.new_cache(self.config)) for r in admitted]
            inputs = [self.prefill_tokens(r, c) for r, c in batch]
        else:
            batch = self.running
            inputs = [[r.output[-1]] for r, _ in batch]
        logits = run_step(
            self.step_fn, [c for _, c in batch], inputs, self.device
        )
        self.steps += 1
        now = time.perf_counter()
        for (request, _), token in zip(batch, logits.argmax(-1), strict=True):
            request.output.append(int(token))
            if request.first_token is None:
                request.first_token = now
            if request.done:
                request.finished = now
        if can_admit:
            for request, cache in batch:
                self.prefilled(request, cache)
            self.running += batch
        self.running = [(r, c) for r, c in self.running if not r.done]
        self.release_finished(batch)

    def prefill_tokens(self, request: Request, cache: object) -> list[int]:
        """Returns the prompt tokens to compute; Section 5.4 skips cached ones."""
        return request.prompt

    def prefilled(self, request: Request, cache: object) -> None:
        """Runs after a request's prefill; Section 5.4 uses it."""

    def release_finished(self, batch: list) -> None:
        """Frees the caches of finished requests; Section 5.3 needs this."""

    def run(self) -> None:
        while self.has_work():
            self.step()
