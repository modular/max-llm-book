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
"""Generated from 01_01_run_and_serve.ipynb by tools/nb2book.py.

Edit the notebook, then run ``pixi run export``.
"""

from __future__ import annotations

import subprocess
import time
import urllib.request


def start_max_serve(
    model: str, *flags: str, port: int = 8000
) -> subprocess.Popen:
    """Starts ``max serve`` and returns once its health check answers.

    Call ``terminate()`` on the returned process to stop the server.
    """
    process = subprocess.Popen(
        ["max", "serve", "--model", model, "--port", str(port), *flags],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    while True:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health")
            return process
        except OSError:
            if process.poll() is not None:
                raise RuntimeError(
                    "max serve exited; run it in a terminal to see why"
                ) from None
            time.sleep(1)
