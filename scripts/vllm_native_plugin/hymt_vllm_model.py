"""Compatibility adapter for the native implementation bundled with vLLM 0.29."""
from vllm.model_executor.models.hunyuan_v1 import HunYuanDenseV1ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader


class HyMT2NativeForCausalLM(HunYuanDenseV1ForCausalLM):
    def load_weights(self, weights):
        # 0.29 removed AutoWeightsLoader(skip_prefixes=...), but its bundled
        # Hunyuan implementation still passes that keyword. Filter lazily to
        # preserve the original tied-embedding rule without copying weights.
        if self.config.tie_word_embeddings:
            weights = ((name, value) for name, value in weights
                       if not name.startswith("lm_head."))
        return AutoWeightsLoader(self).load_weights(weights)
