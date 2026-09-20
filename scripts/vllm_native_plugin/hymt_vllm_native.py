"""Opt-in registration of the native model bundled in vLLM Windows 0.29.

The default 0.29 registry selects TransformersForCausalLM. This experiment
selects the bundled native implementation instead, including its torch.compile
support. Set HYMT_VLLM_NATIVE=1 in both batch and server experiments. Keep this
disabled until output quality and throughput have been checked on the target GPU.
"""
import os


def register():
    if os.environ.get("HYMT_VLLM_NATIVE") != "1":
        return
    from vllm import ModelRegistry
    from importlib.metadata import version
    if not version("vllm").startswith("0.29.0"):
        raise RuntimeError("Hy-MT2 native registration is validated only against vLLM 0.29.0")
    ModelRegistry.register_model(
        "HunYuanDenseV1ForCausalLM",
        "hymt_vllm_model:HyMT2NativeForCausalLM",
    )
