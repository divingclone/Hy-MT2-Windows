"""A stale offline staging tree must not leak weights into the normal release."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_desktop


class ModelFreePackageTests(unittest.TestCase):
    def test_stale_model_zip_and_checkpoint_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            desktop = root / 'desktop'
            payload = desktop / 'src-tauri/resources/payload'
            sources = {
                desktop/'src-tauri/target/release/hymt-desktop.exe': b'exe',
                payload/'models/manifest.json': b'{"schema_version":2}',
                payload/'models/model.zip': b'large weights',
                payload/'models/checkpoint/model.safetensors': b'extracted weights',
                payload/'runtime/vllm/python.exe': b'python',
                payload/'runtime/webview2/msedgewebview2.exe': b'browser',
                payload/'runtime/vllm/Lib/site-packages/torch/lib/cusolverMg64_12.dll': b'optional',
                payload/'runtime/vllm/Lib/site-packages/torch/lib/cusolver64_12.dll': b'required',
                payload/'runtime/vllm/Lib/site-packages/tilelang/__init__.py': b'optional',
            }
            for name in build_desktop.ROOT_DLLS:
                sources[payload/'runtime/vllm'/name] = b'crt'
            for path, data in sources.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            with patch.object(build_desktop, 'DESKTOP', desktop), patch.object(build_desktop, 'PAYLOAD', payload):
                archive = build_desktop.package_portable(root/'output', 'test')
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertEqual([n for n in names if n.startswith('payload/models/')],
                                 ['payload/models/manifest.json'])
                self.assertIn('payload/runtime/vllm/python.exe', names)
                self.assertIn('hymt-desktop.exe', names)
                self.assertFalse(any('webview2' in n or 'tilelang' in n or 'cusolverMg' in n for n in names))
                self.assertIn('payload/runtime/vllm/Lib/site-packages/torch/lib/cusolver64_12.dll',names)
            self.assertEqual((payload/'models/model.zip').read_bytes(), b'large weights')


if __name__ == '__main__':
    unittest.main()
