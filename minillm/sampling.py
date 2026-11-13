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
"""Generated from 01_07_sampling.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

from collections.abc import Callable

from max import functional as F
from max.functional import random
from max.tensor import Tensor


def sample(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Picks a token ID from 1-D ``logits``; ``temperature=0`` is greedy."""
    if temperature == 0:
        return int(F.argmax(logits).item())
    scores, token_ids = logits / temperature, None
    if top_k is not None:
        scores, token_ids = F.top_k(scores, top_k)  # sorted, largest first
    if top_p is not None:
        if token_ids is None:
            token_ids = F.argsort(scores, ascending=False)
            scores = F.gather(scores, token_ids, axis=0)
        probs = F.softmax(scores)
        before = F.cumsum(probs, axis=-1) - probs  # mass of likelier tokens
        scores = F.where(before < top_p, scores, float("-inf"))
    u = random.uniform(
        scores.shape,
        range=(1e-6, 1.0),
        dtype=scores.dtype,
        device=scores.device,
    )
    choice = int(F.argmax(scores - F.log(-F.log(u))).item())
    return choice if token_ids is None else int(token_ids.to_numpy()[choice])


def generate(
    next_logits: Callable[[list[int]], Tensor],
    prompt: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    **sampling: float | None,
) -> list[int]:
    """Generates up to ``max_new_tokens`` tokens after ``prompt``.

    ``next_logits`` maps the tokens so far to the next token's logits.
    Generation stops early at ``eos_token_id``. Returns only the new tokens.
    """
    ids = list(prompt)
    for _ in range(max_new_tokens):
        token = sample(next_logits(ids), **sampling)
        ids.append(token)
        if token == eos_token_id:
            break
    return ids[len(prompt) :]
