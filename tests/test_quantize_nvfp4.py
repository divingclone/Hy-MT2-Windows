"""Quantization workflow guardrails using mock binaries/GGUF; never starts a GPU."""
from contextlib import redirect_stderr, redirect_stdout
from enum import IntEnum
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import quantize_nvfp4 as quant


class Type(IntEnum):
    F32 = 0
    F16 = 1
    BF16 = 30
    NVFP4 = 40


def fields(values):
    return {key: SimpleNamespace(contents=lambda value=value: value) for key, value in values.items()}


def tensor(name, shape, kind=Type.F32, data=None):
    return SimpleNamespace(name=name, shape=shape, tensor_type=kind,
                           data=np.array(data if data is not None else [1.0], dtype=np.float32))


def source_reader():
    return SimpleNamespace(fields=fields({
        'general.architecture': 'hunyuan-dense', 'hunyuan-dense.block_count': 32,
        'hunyuan-dense.embedding_length': 2048, 'hunyuan-dense.feed_forward_length': 6144,
        'hunyuan-dense.attention.head_count': 16, 'hunyuan-dense.attention.head_count_kv': 4,
    }), tensors=[tensor('blk.0.attn_q.weight', [2, 2]), tensor('token_embd.weight', [2, 4])])


def imatrix_reader(name='blk.0.attn_q.weight', values=(1, 2), count=32):
    return SimpleNamespace(fields=fields({'imatrix.datasets': ['calibration.txt'],
                                         'imatrix.chunk_count': 1, 'imatrix.chunk_size': 32}),
                           tensors=[tensor(name + '.in_sum2', [len(values), 1], data=values),
                                    tensor(name + '.counts', [1, 1], data=[count])])


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source.gguf'
        self.source.write_bytes(b'high precision source')
        self.output = self.root / 'candidate.gguf'
        self.binary = self.root / 'bin'
        self.binary.mkdir()
        for name in ('llama-quantize.exe', 'llama-imatrix.exe', 'llama.dll'):
            (self.binary / name).write_bytes(b'mock binary')
        self.reader = source_reader()
        self.matrix_reader = imatrix_reader()
        self.candidate_reader = SimpleNamespace(fields=fields({
            'hymt.quantization.method': 'nvfp4-mse-v1', 'general.quantized_by': 'Hy-MT2-Windows',
        }), tensors=[tensor('blk.0.attn_q.weight', [2, 2], Type.NVFP4)])
        self.gguf = SimpleNamespace(GGMLQuantizationType=Type, GGUFReader=self.read_gguf)
        self.modules = patch.dict(sys.modules, {'gguf': self.gguf,
                                                'repack_hymt_gguf': SimpleNamespace(repack=self.repack)})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.calls = []

    def read_gguf(self, path):
        if Path(path) == self.source:
            return self.reader
        return self.candidate_reader if '.unfused.' in str(path) else self.matrix_reader

    def repack(self, source, output):
        self.assertTrue(source.is_file())
        self.assertFalse(output.exists())
        output.write_bytes(b'fused candidate')
        return {'verified': True}

    def run_binary(self, command, **kwargs):
        self.calls.append(command)
        if '--help' in command:
            return subprocess.CompletedProcess(command, 1, '--nvfp4-mse --nvfp4-mse-full --tensor-type --override-kv --imatrix --output-format --no-ppl --chunks', '')
        if command[0].endswith('llama-imatrix.exe'):
            Path(command[command.index('-o') + 1]).write_bytes(b'matrix')
        else:
            Path(command[-3]).write_bytes(b'nvfp4 candidate')
        return subprocess.CompletedProcess(command, 0)

    def run_main(self, *extra):
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            result = quant.main(['--source', str(self.source), '--output', str(self.output),
                                 '--binary-dir', str(self.binary), *map(str, extra)])
        return result, stdout.getvalue()

    def test_existing_sidecar_and_repack_temporary_are_preserved(self):
        for suffix in ('.gguf', '.unfused.gguf', '.quantization.json', '.quantization.log', '.gguf.tmp'):
            with self.subTest(suffix=suffix):
                path = self.root / ('candidate' + suffix)
                path.write_bytes(b'keep')
                with patch.object(quant.subprocess, 'run') as process:
                    with self.assertRaises(FileExistsError):
                        self.run_main('--dry-run')
                    process.assert_not_called()
                self.assertEqual(path.read_bytes(), b'keep')
                path.unlink()

    def test_output_extension_cannot_collide_with_report(self):
        self.output = self.root / 'candidate.quantization.json'
        with self.assertRaises(SystemExit):
            self.run_main('--dry-run')
        self.assertFalse(self.output.exists())

    def test_requantization_packed_and_split_sources_are_rejected(self):
        for scenario in ('low_bit', 'packed', 'split'):
            with self.subTest(scenario=scenario):
                self.reader = source_reader()
                if scenario == 'low_bit':
                    self.reader.tensors[0].tensor_type = Type.NVFP4
                else:
                    self.reader.fields.update(fields({'hunyuan-dense.hymt_packed_projections': True}
                                                    if scenario == 'packed' else {'split.count': 2}))
                with self.assertRaises(ValueError):
                    self.run_main('--dry-run')

    def test_old_or_crashed_quantizer_is_rejected(self):
        for result in (subprocess.CompletedProcess([], 1, 'old help', ''),
                       subprocess.CompletedProcess([], 255, '--nvfp4-mse --tensor-type --override-kv --imatrix', 'crash')):
            with self.subTest(code=result.returncode), patch.object(quant.subprocess, 'run', return_value=result):
                with self.assertRaisesRegex(ValueError, 'rebuild'):
                    self.run_main('--dry-run')
        self.assertFalse(self.output.exists())

    def test_calibration_plan_accepts_nondefault_chunk_size_without_writes(self):
        calibration = self.root / 'calibration.txt'
        calibration.write_text('example calibration', encoding='utf-8')
        with patch.object(quant.subprocess, 'run', side_effect=self.run_binary):
            result, text = self.run_main('--calibration', calibration, '--context', 384, '--parse-special', '--dry-run')
        self.assertEqual(result, 0)
        command = json.loads(text)['commands'][0]
        self.assertIn('--parse-special', command)
        for flag in ('-c', '-b', '-ub'):
            self.assertEqual(command[command.index(flag) + 1], '384')
        self.assertEqual(command[command.index('-np') + 1], '1')
        self.assertTrue(all('--help' in call for call in self.calls))
        self.assertFalse(self.output.with_suffix('.quantization.json').exists())

    def test_missing_imatrix_executable_fails_before_execution(self):
        (self.binary / 'llama-imatrix.exe').unlink()
        calibration = self.root / 'calibration.txt'
        calibration.write_text('data')
        with patch.object(quant.subprocess, 'run', side_effect=self.run_binary):
            with self.assertRaises(FileNotFoundError):
                self.run_main('--calibration', calibration)
        self.assertTrue(all('--help' in call for call in self.calls))

    def test_external_matrix_records_partial_coverage_without_claiming_checkpoint(self):
        matrix = self.root / 'external.gguf'
        matrix.write_bytes(b'external imatrix')
        with patch.object(quant.subprocess, 'run', side_effect=self.run_binary):
            _, text = self.run_main('--imatrix', matrix, '--dry-run')
        plan = json.loads(text)
        self.assertEqual(plan['imatrix_source'], 'user_supplied_unverified')
        self.assertEqual(plan['imatrix_validation']['unweighted_tensors'], ['token_embd.weight'])

    def test_bad_matrix_does_not_silently_use_unweighted_quantization(self):
        matrix = self.root / 'external.gguf'
        matrix.write_bytes(b'imatrix')
        for reader in (imatrix_reader('other.weight'), imatrix_reader(values=(1, 2, 3)),
                       imatrix_reader(values=(1, float('nan'))), imatrix_reader(count=0)):
            with self.subTest(reader=reader):
                self.matrix_reader = reader
                with patch.object(quant.subprocess, 'run') as process:
                    with self.assertRaises(ValueError):
                        self.run_main('--imatrix', matrix)
                    process.assert_not_called()

    def test_generated_matrix_is_validated_before_quantization(self):
        calibration = self.root / 'calibration.txt'
        calibration.write_text('data')
        self.matrix_reader = imatrix_reader('wrong.weight')
        with patch.object(quant.subprocess, 'run', side_effect=self.run_binary):
            with self.assertRaisesRegex(ValueError, 'no compatible'):
                self.run_main('--calibration', calibration)
        actual = [call for call in self.calls if '--help' not in call]
        self.assertEqual(len(actual), 1)
        report = json.loads(self.output.with_suffix('.quantization.json').read_text())
        self.assertEqual(report['status'], 'failed')

    def test_success_records_actual_types_and_provenance(self):
        with patch.object(quant.subprocess, 'run', side_effect=self.run_binary):
            self.run_main()
        report = json.loads(self.output.with_suffix('.quantization.json').read_text())
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['unfused_tensor_types'], {'NVFP4': 1})
        self.assertEqual(len(report['output_fingerprint']['sha256']), 64)
        self.assertEqual(report['libraries'][0]['filename'], 'llama.dll')
        self.assertIn('general.quantized_by=str:Hy-MT2-Windows', report['commands'][-1])

    def test_generated_matrix_success_records_actual_coverage_and_chunk_count(self):
        calibration = self.root / 'calibration.txt'
        calibration.write_text('data')
        with patch.object(quant.subprocess, 'run', side_effect=self.run_binary):
            self.run_main('--calibration', calibration, '--chunks', 32)
        report = json.loads(self.output.with_suffix('.quantization.json').read_text())
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['imatrix_validation']['chunks'], 1)
        self.assertEqual(report['imatrix_source'], 'generated_from_source')
        self.assertEqual(len(report['imatrix']['sha256']), 64)
        actual = [call for call in self.calls if '--help' not in call]
        self.assertEqual(len(actual), 2)

    def test_full_search_selects_explicit_version_and_verifies_provenance(self):
        self.candidate_reader.fields.update(fields({'hymt.quantization.method': 'nvfp4-mse-full-v2'}))
        with patch.object(quant.subprocess, 'run', side_effect=self.run_binary):
            self.run_main('--scale-search', 'full')
        report = json.loads(self.output.with_suffix('.quantization.json').read_text())
        self.assertEqual(report['method'], 'nvfp4-mse-full-v2')
        self.assertEqual(report['scale_search'], 'full')
        self.assertIn('--nvfp4-mse-full', report['commands'][-1])
        self.assertNotIn('--nvfp4-mse', report['commands'][-1])
        self.assertEqual(report['status'], 'complete')

    def test_full_search_rejects_local_only_binary_before_execution(self):
        result = subprocess.CompletedProcess([], 1, '--nvfp4-mse --tensor-type --override-kv --imatrix', '')
        with patch.object(quant.subprocess, 'run', return_value=result) as process:
            with self.assertRaisesRegex(ValueError, '--nvfp4-mse-full'):
                self.run_main('--scale-search', 'full', '--dry-run')
        self.assertEqual(process.call_count, 1)
        self.assertFalse(self.output.with_suffix('.quantization.json').exists())

    def test_inherited_mse_environment_cannot_override_recorded_version(self):
        with patch.dict(quant.os.environ, {'GGML_NVFP4_QUANTIZE_MSE': '2'}):
            with patch.object(quant.subprocess, 'run', side_effect=self.run_binary) as process:
                _, text = self.run_main('--dry-run')
            self.assertEqual(quant.os.environ['GGML_NVFP4_QUANTIZE_MSE'], '2')
        self.assertEqual(json.loads(text)['method'], 'nvfp4-mse-v1')
        self.assertTrue(all('GGML_NVFP4_QUANTIZE_MSE' not in call.kwargs['env'] for call in process.call_args_list))

    def test_full_search_cannot_accept_local_version_metadata(self):
        with patch.object(quant.subprocess, 'run', side_effect=self.run_binary):
            with self.assertRaisesRegex(ValueError, 'provenance'):
                self.run_main('--scale-search', 'full')
        self.assertFalse(self.output.exists())

    def test_wrong_output_metadata_or_types_cannot_be_marked_complete(self):
        self.candidate_reader.fields = fields({'general.quantized_by': 'Unsloth'})
        with patch.object(quant.subprocess, 'run', side_effect=self.run_binary):
            with self.assertRaisesRegex(ValueError, 'provenance'):
                self.run_main()
        report = json.loads(self.output.with_suffix('.quantization.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(self.output.exists())

    def test_success_exit_without_nvfp4_tensors_is_rejected(self):
        self.candidate_reader.tensors[0].tensor_type = Type.F16
        with patch.object(quant.subprocess, 'run', side_effect=self.run_binary):
            with self.assertRaisesRegex(ValueError, 'no NVFP4'):
                self.run_main()
        report = json.loads(self.output.with_suffix('.quantization.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
