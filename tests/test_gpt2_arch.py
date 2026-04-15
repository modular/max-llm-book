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
"""Smoke tests for the gpt2_arch custom architecture package."""

from typing import Any


def test_architectures_importable() -> None:
    from gpt2_arch import ARCHITECTURES

    assert len(ARCHITECTURES) == 1


def test_architecture_name() -> None:
    from gpt2_arch import ARCHITECTURES

    assert ARCHITECTURES[0].name == "GPT2LMHeadModel"


def test_architecture_task() -> None:
    from gpt2_arch import ARCHITECTURES
    from max.interfaces import PipelineTask

    assert ARCHITECTURES[0].task == PipelineTask.TEXT_GENERATION


def test_config_implements_arch_config() -> None:
    from gpt2_arch.model_config import GPT2ArchConfig
    from max.pipelines.lib.interfaces import ArchConfig, ArchConfigWithKVCache

    assert issubclass(GPT2ArchConfig, ArchConfig)
    assert issubclass(GPT2ArchConfig, ArchConfigWithKVCache)


def _make_weights(arrays: dict[str, Any]) -> dict[str, Any]:
    """Wrap numpy arrays as minimal Weights-protocol objects for adapter tests."""
    import numpy as np
    from max.graph.weights import WeightData

    class _FakeWeight:
        def __init__(self, arr: np.ndarray, name: str) -> None:
            self._wd = WeightData.from_numpy(np.ascontiguousarray(arr), name)

        def data(self) -> WeightData:
            return self._wd

    return {name: _FakeWeight(arr, name) for name, arr in arrays.items()}


def test_weight_adapter_transposes_conv1d_weights() -> None:
    import numpy as np
    from gpt2_arch.weight_adapters import convert_safetensor_state_dict

    # GPT-2 Conv1D weight is stored as [in, out]; MAX Linear expects [out, in]
    fake_weight = np.ones((768, 2304))  # c_attn: [in=768, out=3*768]
    sd = _make_weights({"transformer.h.0.attn.c_attn.weight": fake_weight})
    result = convert_safetensor_state_dict(sd)
    key = "transformer.h.0.attn.c_attn.weight"
    assert tuple(np.from_dlpack(result[key]).shape) == (2304, 768)


def test_weight_adapter_fills_lm_head_from_wte() -> None:
    import numpy as np
    from gpt2_arch.weight_adapters import convert_safetensor_state_dict

    wte = np.ones((50257, 768))
    sd = _make_weights({"transformer.wte.weight": wte})
    result = convert_safetensor_state_dict(sd)
    # lm_head.weight should be added (tied from wte) as [out=50257, in=768]
    assert "lm_head.weight" in result
    assert tuple(np.from_dlpack(result["lm_head.weight"]).shape) == (50257, 768)


def test_gpt2_lm_head_model_imports() -> None:
    from gpt2_arch.gpt2 import MaxGPT2LMHeadModel  # noqa: F401


def test_gpt2_lm_head_model_has_expected_submodules() -> None:
    import types

    import max.experimental.functional as F
    from gpt2_arch.gpt2 import MaxGPT2LMHeadModel

    cfg = types.SimpleNamespace(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=2,
        n_head=12,
        n_inner=None,
        layer_norm_epsilon=1e-5,
    )
    with F.lazy():
        mod = MaxGPT2LMHeadModel(cfg)  # type: ignore[arg-type]

    # Top-level: transformer body + lm_head
    assert hasattr(mod, "transformer")
    assert hasattr(mod, "lm_head")
    # Body contains the GPT-2 components
    assert hasattr(mod.transformer, "wte")
    assert hasattr(mod.transformer, "wpe")
    assert hasattr(mod.transformer, "h")
    assert len(mod.transformer.h) == 2
    assert hasattr(mod.transformer, "ln_f")


def test_pipeline_model_is_pipeline_model_with_kv_cache() -> None:
    from gpt2_arch.model import GPT2PipelineModel
    from max.pipelines.lib import PipelineModelWithKVCache

    assert issubclass(GPT2PipelineModel, PipelineModelWithKVCache)


def test_pipeline_model_get_kv_params_returns_params() -> None:
    """get_kv_params is a classmethod; test it can be imported and has right signature."""
    from gpt2_arch.model import GPT2PipelineModel

    assert hasattr(GPT2PipelineModel, "get_kv_params")
    assert hasattr(GPT2PipelineModel, "calculate_max_seq_len")
    assert hasattr(GPT2PipelineModel, "execute")
    assert hasattr(GPT2PipelineModel, "prepare_initial_token_inputs")
    assert hasattr(GPT2PipelineModel, "prepare_next_token_inputs")


def test_pixi_toml_has_serve_task() -> None:
    """Verify the pixi.toml defines a 'serve' task pointing at max serve."""
    import pathlib

    import tomllib  # type: ignore[import-not-found]

    pixi_path = pathlib.Path(__file__).parent.parent / "pixi.toml"
    with pixi_path.open("rb") as fh:
        config = tomllib.load(fh)

    tasks = config.get("tasks", {})
    assert "serve" in tasks, "pixi.toml must define a 'serve' task"
    serve_cmd = tasks["serve"]
    if isinstance(serve_cmd, dict):
        serve_cmd = serve_cmd.get("cmd", "")
    assert "max serve" in serve_cmd
    assert "--custom-architectures" in serve_cmd
    assert "gpt2_arch" in serve_cmd
    assert "--model" in serve_cmd
    assert "gpt2" in serve_cmd
