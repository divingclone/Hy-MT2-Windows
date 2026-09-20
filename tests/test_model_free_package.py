"""A stale offline staging tree must not leak weights into the normal release."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_desktop
from runtime_filter import excluded
from audit_runtime_size import audit_archive
from package_windows import runtime_problems, sha256


class ModelFreePackageTests(unittest.TestCase):
    def test_custom_torch_requires_validation_and_matching_binaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root/'runtime/vllm'
            torch = runtime/'Lib/site-packages/torch'
            (torch/'lib').mkdir(parents=True)
            (torch/'cuda').mkdir()
            (torch/'lib/torch_cuda.dll').write_bytes(b'custom backend')
            (torch/'lib/torch_cpu.dll').write_bytes(b'original CPU')
            (torch/'cuda/__init__.py').write_bytes(b'actual architectures')
            (runtime/'hymt-runtime.json').write_text(json.dumps({
                'vllm':'0.29.0+cu132', 'torch':'2.11.0+cu130',
                'torch_custom_build':'hymt-cuda-delay-v1'}))
            record = {'profile':'hymt-cuda-delay-v1', 'binary_compatibility_checked':True,
                      'inference_validated':False,
                      'custom_dll_sha256':sha256(torch/'lib/torch_cuda.dll'),
                      'cuda_python_sha256':sha256(torch/'cuda/__init__.py'),
                      'build_provenance':{'base_libraries':{
                          'torch_cpu.dll':sha256(torch/'lib/torch_cpu.dll')}}}
            marker = torch/'hymt-build.json'
            self.assertTrue(runtime_problems(root))  # missing marker
            marker.write_text(json.dumps(record))
            self.assertTrue(runtime_problems(root))  # not yet GPU-validated
            record['inference_validated'] = True
            marker.write_text(json.dumps(record))
            self.assertEqual(runtime_problems(root), [])
            (torch/'lib/torch_cpu.dll').write_bytes(b'different CPU')
            self.assertTrue(runtime_problems(root))

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
                payload/'runtime/vllm/hymt-runtime.json': b'{"vllm":"0.29.0+cu132","torch":"2.11.0+cu130"}',
                payload/'runtime/webview2/msedgewebview2.exe': b'browser',
                payload/'runtime/vllm/Lib/site-packages/torch/lib/cusolverMg64_12.dll': b'optional',
                payload/'runtime/vllm/Lib/site-packages/torch/lib/cusolver64_12.dll': b'required',
                payload/'runtime/vllm/Lib/site-packages/torch/lib/cudnn_engines_precompiled64_9.dll': b'full-only',
                payload/'runtime/vllm/Lib/site-packages/torch/lib/cudnn64_9.dll': b'dispatcher',
                payload/'runtime/vllm/Lib/site-packages/torch/lib/cudnn_graph64_9.dll': b'graph',
                payload/'runtime/vllm/Lib/site-packages/torch/lib/cudnn_engines_runtime_compiled64_9.dll': b'jit',
                payload/'runtime/vllm/Lib/site-packages/tilelang/__init__.py': b'optional',
                payload/'runtime/vllm/Lib/site-packages/cv2/opencv_videoio_ffmpeg500_64.dll': b'video',
                payload/'runtime/vllm/Lib/site-packages/cv2/cv2.pyd': b'imported by vllm',
                payload/'runtime/vllm/Lib/site-packages/samples/jupyter/demo.ipynb': b'video sample',
                payload/'runtime/vllm/Lib/site-packages/torch/lib/torch_cpu.lib': b'msvc linking',
                payload/'runtime/vllm/Lib/site-packages/numpy/_core/tests/data/example.npy': b'test data',
                payload/'runtime/vllm/Lib/site-packages/numpy/testing/__init__.py': b'imported helper',
                payload/'runtime/python/tcl/tk8.6/tk.tcl': b'tk',
            }
            for name in build_desktop.ROOT_DLLS:
                sources[payload/'runtime/vllm'/name] = b'crt'
            for path, data in sources.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            with patch.object(build_desktop, 'DESKTOP', desktop), patch.object(build_desktop, 'PAYLOAD', payload), patch.object(build_desktop, 'patch_release_runtime') as release_patch:
                archive = build_desktop.package_portable(root/'output', 'test')
                release_patch.assert_called_once_with(root/'output/HyMT-test-windows-x64-portable/payload')
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertEqual([n for n in names if n.startswith('payload/models/')],
                                 ['payload/models/manifest.json'])
                self.assertIn('payload/runtime/vllm/python.exe', names)
                self.assertIn('hymt-desktop.exe', names)
                self.assertFalse(any('webview2' in n or 'tilelang' in n or 'cusolverMg' in n for n in names))
                self.assertIn('payload/runtime/vllm/Lib/site-packages/torch/lib/cusolver64_12.dll',names)
                self.assertNotIn('payload/runtime/vllm/Lib/site-packages/torch/lib/cudnn_engines_precompiled64_9.dll',names)
                for library in ('cudnn64_9.dll','cudnn_graph64_9.dll','cudnn_engines_runtime_compiled64_9.dll'):
                    self.assertIn('payload/runtime/vllm/Lib/site-packages/torch/lib/'+library,names)
                self.assertNotIn('payload/runtime/vllm/Lib/site-packages/cv2/cv2.pyd', names)
                self.assertIn('payload/runtime/vllm/Lib/site-packages/numpy/testing/__init__.py', names)
                self.assertFalse(any(excluded(n.removeprefix('payload/')) for n in names))
                self.assertFalse(audit_archive(archive)['removed'])
            self.assertEqual((payload/'models/model.zip').read_bytes(), b'large weights')

    def test_runtime_filter_keeps_compilers_headers_licenses_and_import_helpers(self):
        retained = (
            'runtime/vllm/include/Python.h',
            'runtime/vllm/libs/python312.lib',
            'runtime/vllm/Lib/site-packages/torch/include/torch/extension.h',
            'runtime/vllm/Lib/site-packages/triton/backends/nvidia/lib/cuda.lib',
            'runtime/vllm/Lib/site-packages/triton/runtime/tcc/tcc.exe',
            'runtime/vllm/Lib/site-packages/flashinfer/data/aot/sampling/sampling.dll',
            'runtime/vllm/Lib/site-packages/numpy/testing/__init__.py',
            'runtime/vllm/Lib/site-packages/sympy/testing/__init__.py',
            'runtime/vllm/Lib/site-packages/torch/testing/_internal/common_utils.py',
            'runtime/vllm/Lib/site-packages/pycountry/databases/iso639-3.json',
            'runtime/vllm/Lib/site-packages/pycountry-24.6.1.dist-info/licenses/LICENSE.txt',
            'runtime/python/DLLs/_ctypes.pyd',
            'runtime/python/DLLs/_ssl.pyd',
            'docs/tests/example.txt',
        )
        for path in retained:
            with self.subTest(path=path):
                self.assertFalse(excluded(path))
                self.assertFalse(excluded(path.replace('/', '\\')))

    def test_archive_audit_counts_only_new_runtime_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)/'baseline.zip'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
                zipped.writestr('payload/runtime/vllm/Lib/site-packages/samples/demo.ipynb', b'x'*1000)
                zipped.writestr('payload/runtime/vllm/Lib/site-packages/torch/lib/torch_cuda.dll', b'keep')
                zipped.writestr('payload/runtime/webview2/msedgewebview2.exe', b'explicit offline package')
                zipped.writestr('hymt-desktop.exe', b'exe')
            report = audit_archive(archive)
            self.assertEqual(report['removed_bytes'], 1000)
            self.assertEqual(len(report['removed']), 1)
            self.assertLess(report['removed_compressed_bytes'], 1000)

    def test_skip_stage_rejects_unvalidated_runtime_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root/'payload/runtime/vllm/hymt-runtime.json'
            marker.parent.mkdir(parents=True)
            marker.write_text('{"vllm":"new-version","torch":"2.11.0+cu130"}', encoding='utf-8')
            with patch.object(build_desktop, 'PAYLOAD', root/'payload'):
                with self.assertRaisesRegex(ValueError, 'revalidation'):
                    build_desktop.package_portable(root/'output', 'test')
            self.assertFalse((root/'output').exists())


if __name__ == '__main__':
    unittest.main()
