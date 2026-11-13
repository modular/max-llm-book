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
"""Keeps MAX's compiled-model cache in a temporary folder for the session."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def temporary_max_cache() -> Iterator[None]:
    # Every notebook compiles models, and cached compiled models are large;
    # a full run would otherwise leave gigabytes in the environment's cache.
    cache = tempfile.mkdtemp(prefix="max-llm-book-cache-")
    previous = os.environ.get("MODULAR_MAX_CACHE_DIR")
    os.environ["MODULAR_MAX_CACHE_DIR"] = cache
    yield
    if previous is None:
        os.environ.pop("MODULAR_MAX_CACHE_DIR", None)
    else:
        os.environ["MODULAR_MAX_CACHE_DIR"] = previous
    shutil.rmtree(cache, ignore_errors=True)
