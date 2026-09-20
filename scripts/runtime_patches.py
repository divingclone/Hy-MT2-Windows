"""Small, hash-pinned release-only import changes; never edit the source venv."""
import hashlib
from pathlib import Path

VISION_ATTENTION = 'runtime/vllm/Lib/site-packages/vllm/model_executor/layers/attention/mm_encoder_attention.py'
BEFORE_SHA256 = '8dffe42a5e6bc49eac62ca22b677833541c1338d9f4b91abbb4fe19d9d04b031'
AFTER_SHA256 = '2441425532cd0ca156b2ce7a82a659ad065b1a9749229b7c580f810b32c0030d'


def patch_release_runtime(root):
    """Defer the vision-only FA import, allowing text releases to omit FA2.

    The selected Triton/CUTLASS/Marlin implementation is untouched. Preserve
    upstream's ImportError if a caller actually requests the absent backend.
    Both staged and already-patched payloads are accepted; other sources fail.
    """
    path = Path(root)/VISION_ATTENTION
    original = path.read_bytes()
    source = original.decode('utf-8').replace('\r\n', '\n')
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == AFTER_SHA256:
        return
    if digest != BEFORE_SHA256:
        raise ValueError('Unrecognized vLLM vision attention source; revalidate release patch')
    source = source.replace('from vllm.v1.attention.backends.fa_utils import get_flash_attn_version\n', '')
    source = source.replace('        self._fa_version = (\n',
        '        # HyMT: import the optional backend only when selected by a vision model.\n'
        '        if self.is_flash_attn_backend:\n'
        '            from vllm.v1.attention.backends.fa_utils import get_flash_attn_version\n\n'
        '        self._fa_version = (\n')
    if hashlib.sha256(source.encode()).hexdigest() != AFTER_SHA256:
        raise ValueError('vLLM release patch result differs from the reviewed source')
    if b'\r\n' in original:
        source = source.replace('\n', '\r\n')
    path.write_bytes(source.encode('utf-8'))
