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
# The first cell of every chapter notebook. `pixi run bootstrap` copies the
# code below the marker into each notebook; edit it here, not in a notebook.
# --- cell ---
# Setup: installs MAX and this book's `minillm` package with uv if they are
# missing, as on Colab or Kaggle (turn on Internet in Kaggle's settings
# first). In the book's pixi environment, this cell does nothing.
import importlib.util
import os
import shutil
import subprocess
import sys

BOOK_REF = "main"  # Branch or tag of github.com/modular/max-llm-book.
BOOK_SRC = os.environ.get(
    "MAX_LLM_BOOK_SRC",
    f"git+https://github.com/modular/max-llm-book@{BOOK_REF}",
)


def uv_install(*packages: str) -> None:
    if shutil.which("uv") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "uv"], check=True
        )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "-q",
            "--prerelease",
            "allow",
            "--index-strategy",
            "unsafe-best-match",
            "--extra-index-url",
            "https://whl.modular.com/nightly/simple/",
            *packages,
        ],
        check=True,
    )


if importlib.util.find_spec("max") is None:
    uv_install("max[all]")
if importlib.util.find_spec("minillm") is None:
    uv_install(BOOK_SRC)
# Cells run `max` and `mojo` as commands; find them next to this Python even
# when the environment isn't activated.
os.environ["PATH"] = (
    os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
)
# MAX's matmul needs an NVIDIA GPU from the Ampere generation (compute
# capability 8.0) or newer. Hide older ones, such as Colab's T4, so MAX uses
# the CPU instead of failing.
if shutil.which("nvidia-smi"):
    caps = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.split()
    usable = [str(i) for i, cap in enumerate(caps) if float(cap) >= 8.0]
    if len(usable) < len(caps):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(usable)
        print(
            f"Using {len(usable)} of {len(caps)} GPUs; MAX needs compute capability 8.0 or newer."
        )
