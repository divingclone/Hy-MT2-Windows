"""Pure decision/fixture tests. No nvidia-smi, CUDA, models or inference execute."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gpu_config as config


BUILD_INFO = {
    "architectures": ["75", "80", "86", "89", "120a"],
    "ptx_architectures": [], "minimum_driver": "580.88",
    "nvfp4_compute_capabilities": ["12.0"], "cuda_toolkit": "13.0",
}


def gpu(index=0, capability="12.0", free=30000, driver="596.36"):
    return {"index": index, "uuid": f"GPU-unit-test-{index}", "name": f"NVIDIA Test {index}",
            "compute_capability": capability, "total_memory_mib": 32768,
            "free_memory_mib": free, "driver_version": driver}


class ProbeTests(unittest.TestCase):
    def test_csv_units_quotes_and_multiple_devices(self):
        devices = config.parse_smi_csv('0, GPU-abc, "NVIDIA, test", 7.5, 4096, 3072, 580.88\r\n'
                                       '1, GPU-def, RTX test, 12.0, 32768, 30000, 596.36\n')
        self.assertEqual(devices[0]["name"], "NVIDIA, test")
        self.assertEqual(devices[0]["free_memory_mib"], 3072)
        self.assertEqual(devices[1]["compute_capability"], "12.0")

    def test_bad_probe_data_rejected(self):
        for data in ("", "No devices were found", "0, GPU-a, Test, 7.5, N/A, N/A, 580.88",
                     "0, GPU-a, Test, 7.5, 4096, 5000, 580.88",
                     "0, GPU-a, Test, 7.5, 4096, nan, 580.88",
                     "0, GPU-a, Test, unknown, 4096, 3000, 580.88",
                     "0, GPU-a, Test, 7.5, 4096, 3000, unknown"):
            with self.subTest(data=data), self.assertRaises(config.ConfigError):
                config.parse_smi_csv(data)

    def test_missing_driver_and_timeout_are_actionable(self):
        with patch.object(config.shutil, "which", return_value=None), patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(config.ConfigError, "nvidia-smi"):
                config.detect_gpus()
        with patch.object(config.subprocess, "run", side_effect=subprocess.TimeoutExpired("mock", 1)):
            with self.assertRaisesRegex(config.ConfigError, "超时"):
                config.detect_gpus("mock-smi")
        result = subprocess.CompletedProcess([], 9, "", "driver/library mismatch")
        with patch.object(config.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(config.ConfigError, "驱动"):
                config.detect_gpus("mock-smi")

    def test_query_is_bounded_and_read_only(self):
        result = subprocess.CompletedProcess([], 0, "0, GPU-a, Test, 8.6, 12288, 9000, 596.36\n", "")
        with patch.object(config.subprocess, "run", return_value=result) as run:
            self.assertEqual(config.detect_gpus("mock-smi", timeout=3)[0]["index"], 0)
        args, kwargs = run.call_args
        self.assertIn("--format=csv,noheader,nounits", args[0])
        self.assertEqual(kwargs["timeout"], 3)
        self.assertNotIn("shell", kwargs)


class MemoryTests(unittest.TestCase):
    def plan(self, **kwargs):
        return config.memory_plan(**dict(mode="batch", free_mib=30000, model_bytes=1024*config.MIB, **kwargs))

    def test_small_card_free_three_gib(self):
        result = config.memory_plan(mode="batch", free_mib=3072, model_bytes=1024*config.MIB)
        self.assertEqual(result["parallel"], 8)
        self.assertEqual(result["ubatch"], 256)
        self.assertEqual(result["budget"]["workspace_reserve_mib"], 1024)
        self.assertEqual(result["budget"]["safety_margin_mib"], 256)
        self.assertGreaterEqual(result["budget"]["remaining_after_estimate_mib"], 0)

    def test_batch_and_server_caps(self):
        self.assertEqual(self.plan()["parallel"], 256)
        server = config.memory_plan(mode="server", free_mib=30000, model_bytes=1024*config.MIB)
        self.assertEqual(server["parallel"], 128)
        self.assertEqual(server["ubatch"], 512)
        with self.assertRaises(config.ConfigError):
            config.memory_plan(mode="server", free_mib=30000, model_bytes=1024*config.MIB, parallel=129)

    def test_context_is_rounded_and_scales_kv(self):
        self.assertEqual(self.plan(context=1024)["budget"]["kv_per_slot_mib"], 64)
        self.assertEqual(self.plan(context=1025)["budget"]["kv_per_slot_mib"], 80)
        self.assertEqual(self.plan(context=2048)["budget"]["kv_per_slot_mib"], 128)
        self.assertLess(self.plan(context=2048)["parallel"], self.plan()["parallel"])

    def test_explicit_values_are_preserved_and_checked(self):
        result = self.plan(parallel=3, ubatch=17)
        self.assertEqual((result["parallel"], result["ubatch"]), (3, 17))
        self.assertEqual(self.plan(ubatch=32)["parallel"], 32)
        with self.assertRaisesRegex(config.ConfigError, "ubatch"):
            self.plan(parallel=64, ubatch=32)
        with self.assertRaisesRegex(config.ConfigError, "显式parallel"):
            config.memory_plan(mode="batch", free_mib=3072, model_bytes=1024*config.MIB, parallel=32)

    def test_impossible_values_do_not_produce_zero_parallel(self):
        for kwargs in ({"parallel": 0}, {"parallel": 257}, {"context": 0}, {"ubatch": 0},
                       {"parallel": True}, {"parallel": 256, "context": config.INT32_MAX}):
            with self.subTest(kwargs=kwargs), self.assertRaises(config.ConfigError):
                self.plan(**kwargs)
        with self.assertRaisesRegex(config.ConfigError, "空闲显存"):
            config.memory_plan(mode="batch", free_mib=1024, model_bytes=1024*config.MIB)

    def test_explicit_large_ubatch_reserves_extra_workspace(self):
        result = self.plan(parallel=1, ubatch=4096)
        self.assertEqual(result["batch"], 4096)
        self.assertEqual(result["budget"]["workspace_reserve_mib"], 4096)


    def test_quantized_kv_includes_block_scales_and_independent_types(self):
        for key, value, expected_k, expected_v in (("q8_0", "q8_0", 17, 17),
                                                   ("q4_0", "q4_0", 9, 9),
                                                   ("f16", "q8_0", 32, 17),
                                                   ("q8_0", "q4_0", 17, 9)):
            with self.subTest(key=key, value=value):
                result = self.plan(parallel=8, cache_type_k=key, cache_type_v=value)
                budget = result["budget"]
                self.assertEqual((result["cache_type_k"], result["cache_type_v"]), (key, value))
                self.assertEqual((budget["k_per_slot_mib"], budget["v_per_slot_mib"]), (expected_k, expected_v))
                self.assertEqual(budget["kv_total_mib"], 8 * (expected_k + expected_v))
                self.assertEqual(budget["k_total_mib"] + budget["v_total_mib"], budget["kv_total_mib"])
        rounded = self.plan(context=1025, cache_type_k="q8_0", cache_type_v="q8_0")
        self.assertEqual(rounded["budget"]["kv_per_slot_mib"], 42.5)

    def test_quantized_cache_changes_admission_budget(self):
        for cache_type, expected_parallel in (("f16", 8), ("q8_0", 16), ("q4_0", 32)):
            result = config.memory_plan(mode="batch", free_mib=3072, model_bytes=1024*config.MIB,
                                        cache_type_k=cache_type, cache_type_v=cache_type)
            self.assertEqual(result["parallel"], expected_parallel)
        for kwargs in ({"cache_type_k": "fp8"}, {"cache_type_v": "q4_k"}):
            with self.assertRaisesRegex(config.ConfigError, "cache-type"):
                self.plan(**kwargs)


class CacheCompatibilityTests(unittest.TestCase):
    def test_f16_requires_current_binary_and_passes_explicit_cache_types(self):
        self.assertEqual(config.cache_type_args(), ['--cache-type-k', 'f16', '--cache-type-v', 'f16'])
        supported = subprocess.CompletedProcess([], 0, '--cache-type-k TYPE\n--cache-type-v TYPE', '')
        with patch.object(config.subprocess, "run", return_value=supported) as run:
            config.ensure_cache_type_support("hy-batch.exe", "f16", "f16")
        run.assert_called_once()

    def test_quantized_cache_requires_both_flags_without_loading_a_model(self):
        supported = subprocess.CompletedProcess([], 0, "--cache-type-k TYPE\n--cache-type-v TYPE", "")
        with patch.object(config.subprocess, "run", return_value=supported) as run:
            config.ensure_cache_type_support("new-hy-batch.exe", "q8_0", "q4_0", env={"PATH": "test"})
        self.assertEqual(run.call_args.args[0], ["new-hy-batch.exe", "--help"])
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertEqual(run.call_args.kwargs["env"], {"PATH": "test"})
        self.assertEqual(config.cache_type_args("q8_0", "f16"),
                         ["--cache-type-k", "q8_0", "--cache-type-v", "f16"])

    def test_old_binary_and_failed_probe_are_actionable(self):
        for result in (subprocess.CompletedProcess([], 0, "old help", ""),
                       subprocess.CompletedProcess([], 0, "--cache-type-k TYPE", ""),
                       subprocess.CompletedProcess([], 1, "--cache-type-k --cache-type-v", "failed")):
            with patch.object(config.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(config.ConfigError, "build-source.ps1"):
                    config.ensure_cache_type_support("old.exe", "q8_0", "q8_0")
        with patch.object(config.subprocess, "run", side_effect=subprocess.TimeoutExpired("mock", 15)):
            with self.assertRaisesRegex(config.ConfigError, "build-source.ps1"):
                config.ensure_cache_type_support("old.exe", "q8_0", "q8_0")


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hy-gpu-config-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "可移动安装"
        self.root.mkdir()
        self.make_binary(self.root / "bin")
        model_dir = self.root / "models"
        model_dir.mkdir()
        for filename in (config.MODEL_FILES["fast"][0], config.MODEL_FILES["official"][0]):
            (model_dir / filename).write_bytes(b"fixture GGUF availability only")
        (model_dir / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'files': {
            profile: {'filename': names[0], 'size_bytes': (model_dir / names[0]).stat().st_size,
                      'sha256': hashlib.sha256((model_dir / names[0]).read_bytes()).hexdigest()}
            for profile, names in config.MODEL_FILES.items()}}), encoding='utf-8')
        self.detect = patch.object(config, "detect_gpus", return_value=[gpu()]).start()
        self.addCleanup(patch.stopall)

    def make_binary(self, directory, metadata=True):
        directory.mkdir(parents=True, exist_ok=True)
        for filename in config.MODE_EXECUTABLES.values():
            (directory / filename).write_bytes(b"never executed")
        if metadata:
            (directory / "build-info.json").write_text(json.dumps(BUILD_INFO), encoding="utf-8")

    def test_packaged_paths_are_absolute_and_independent_of_cwd(self):
        result = config.resolve_config(self.root)
        self.assertEqual(result["binary_dir"], str(self.root / "bin"))
        self.assertEqual(result["profile"], "fast")
        self.assertTrue(Path(result["model"]).is_absolute())
        self.assertEqual(result["environment"]["CUDA_VISIBLE_DEVICES"], gpu()["uuid"])
        self.assertEqual(result["device_index"], 0)
        self.assertTrue(result["gpu_prefix"])
        self.assertEqual(result["sampling_threads"], 1)

    def test_old_card_uses_official_and_rejects_fast(self):
        self.detect.return_value = [gpu(capability="7.5", free=3072)]
        self.assertEqual(config.resolve_config(self.root)["profile"], "official")
        with self.assertRaisesRegex(config.ConfigError, "NVFP4"):
            config.resolve_config(self.root, profile="fast")

    def test_120a_is_exact_not_forward_compatible(self):
        self.assertIsNotNone(config.architecture_match("12.0", BUILD_INFO))
        for capability in ("12.1", "13.0", "9.0", "6.1"):
            self.assertIsNone(config.architecture_match(capability, BUILD_INFO))
        explicit_ptx = dict(BUILD_INFO, ptx_architectures=["120"])
        self.assertEqual(config.architecture_match("12.1", explicit_ptx)["kind"], "ptx_forward")

    def test_auto_skips_unsupported_and_selects_most_free_compatible_gpu(self):
        self.detect.return_value = [gpu(0, "12.1", 31000), gpu(1, "8.6", 12000), gpu(2, "8.9", 16000)]
        result = config.resolve_config(self.root)
        self.assertEqual(result["gpu_index"], 2)
        self.assertEqual(result["profile"], "official")

    def test_explicit_index_or_uuid_is_respected(self):
        self.detect.return_value = [gpu(0, free=30000), gpu(1, "8.6", 12000)]
        for selected in (1, "1", "GPU-unit-test-1"):
            self.assertEqual(config.resolve_config(self.root, gpu=selected)["gpu_index"], 1)
        with self.assertRaisesRegex(config.ConfigError, "指定GPU"):
            config.resolve_config(self.root, gpu="GPU-missing")

    def test_driver_floor_comes_from_metadata(self):
        self.detect.return_value = [gpu(driver="580.65")]
        with self.assertRaisesRegex(config.ConfigError, "580.88"):
            config.resolve_config(self.root)
        self.detect.return_value = [gpu(driver="580.88")]
        self.assertTrue(config.resolve_config(self.root)["ok"])

    def test_installed_bin_is_authoritative_and_build_override_is_explicit(self):
        portable = self.root / "build/portable/bin"
        self.make_binary(portable)
        self.assertEqual(config.resolve_config(self.root)["binary_dir"], str(self.root / "bin"))
        self.assertEqual(config.resolve_config(self.root, binary_dir="build/portable/bin")["binary_dir"], str(portable))

    def test_metadata_required_even_when_an_experimental_build_exists(self):
        (self.root / "bin/build-info.json").unlink()
        with self.assertRaisesRegex(config.ConfigError, "build-info"):
            config.resolve_config(self.root)
        legacy = self.root / "build/optimized/bin"
        self.make_binary(legacy, metadata=False)
        with self.assertRaises(config.ConfigError):
            config.resolve_config(self.root)
        with self.assertRaises(config.ConfigError):
            config.resolve_config(self.root, binary_dir=legacy)

    def test_missing_current_nvfp4_requires_install_instead_of_switching_model(self):
        (self.root / "models" / config.MODEL_FILES["fast"][0]).unlink()
        with self.assertRaisesRegex(config.ConfigError, "setup-model.cmd"):
            config.resolve_config(self.root)
        with self.assertRaisesRegex(config.ConfigError, "模型"):
            config.resolve_config(self.root, profile="fast")

    def test_manifest_selects_current_payload_and_rejects_wrong_size_or_path(self):
        path = self.root / 'models/manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        item = manifest['files']['fast']
        item['filename'] = 'Hy-MT-calibrated.gguf'
        (self.root / 'models' / item['filename']).write_bytes(b'new calibrated payload')
        item['size_bytes'] = 22
        item['sha256'] = hashlib.sha256(b'new calibrated payload').hexdigest()
        path.write_text(json.dumps(manifest), encoding='utf-8')
        self.assertEqual(Path(config.resolve_config(self.root)['model']).name, item['filename'])
        item['size_bytes'] += 1
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(config.ConfigError, '大小'):
            config.resolve_config(self.root)
        item['filename'] = '../escape.gguf'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(config.ConfigError, 'manifest'):
            config.resolve_config(self.root)

    def test_same_size_same_mtime_replacement_fails_full_hash_validation(self):
        result = config.resolve_config(self.root)
        path = Path(result['model'])
        before = path.stat()
        path.write_bytes(b'X' * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(path.stat().st_size, before.st_size)
        self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
        with self.assertRaisesRegex(config.ConfigError, 'SHA-256'):
            config.resolve_config(self.root)

    def test_manifest_requires_a_valid_sha256(self):
        path = self.root / 'models/manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        for value in (None, '', 17, True, 'f' * 63, 'f' * 65, 'g' * 64, 'f' * 64 + '\n'):
            with self.subTest(value=value):
                manifest['files']['fast']['sha256'] = value
                path.write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaisesRegex(config.ConfigError, 'manifest'):
                    config.resolve_config(self.root)
        del manifest['files']['fast']['sha256']
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(config.ConfigError, 'manifest'):
            config.resolve_config(self.root)

    def test_explicit_model_override_does_not_require_manifest_identity(self):
        path = self.root / 'models' / config.MODEL_FILES['fast'][0]
        path.write_bytes(b'user selected model with a different identity')
        (self.root / 'models/manifest.json').write_text('invalid manifest', encoding='utf-8')
        self.assertEqual(config.resolve_config(self.root, model_override=path)['model'], str(path))

    def test_server_and_model_override(self):
        chosen = self.root / "指定模型.gguf"
        chosen.write_bytes(b"custom existing model")
        result = config.resolve_config(self.root, mode="server", model_override=chosen.name)
        self.assertEqual(result["model"], str(chosen))
        self.assertEqual(result["model_override"], str(chosen))
        self.assertEqual(result["budget"]["model_file_bytes"], chosen.stat().st_size)
        self.assertFalse(result["gpu_prefix"])
        self.assertEqual(result["parallel"], 128)
        with self.assertRaisesRegex(config.ConfigError, "指定模型"):
            config.resolve_config(self.root, model_override="missing.gguf")

    def test_cache_selection_survives_resolution_and_json_cli(self):
        result = config.resolve_config(self.root, cache_type_k="q8_0", cache_type_v="q4_0")
        self.assertEqual(result["budget"]["kv_per_slot_mib"], 26)
        self.assertTrue(any("混合K/V" in warning for warning in result["warnings"]))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = config.main(["--root", str(self.root), "--json", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["budget"]["kv_per_slot_mib"], 34)

    def test_json_stdout_is_ascii_and_roundtrips_chinese_paths(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = config.main(["--root", str(self.root), "--json"])
        self.assertEqual(status, 0)
        raw = output.getvalue()
        self.assertTrue(raw.isascii())
        self.assertEqual(json.loads(raw)["root"], str(self.root))
        self.detect.side_effect = config.ConfigError("驱动未安装")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = config.main(["--root", str(self.root), "--json"])
        self.assertEqual(status, 2)
        self.assertTrue(output.getvalue().isascii())
        self.assertEqual(json.loads(output.getvalue())["error"], "驱动未安装")


if __name__ == "__main__":
    unittest.main()
