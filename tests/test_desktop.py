import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import desktop_bridge as bridge
import desktop_update as updater


class DesktopConfigurationTests(unittest.TestCase):
    def test_config_rejects_boolean_fraction_and_unbounded_allocation(self):
        for values in ({'parallel': True}, {'parallel': 1.5}, {'parallel': 257},
                       {'context': 0}, {'context': 32769}, {'port': 1}, {'port': 65536},
                       {'ubatch': -1}, {'cache': 'bogus'}, {'profile': '../../bad'}, {'gpu': []},
                       {'memoryPercent': 9}, {'memoryPercent': 101}, {'memoryPercent': True}, {'memoryPercent': 30.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                bridge.validate_settings(values)

    def test_defaults_and_explicit_settings_are_preserved(self):
        self.assertEqual(bridge.validate_settings({})['parallel'], 0)
        self.assertEqual(bridge.validate_settings({})['context'], 2048)
        self.assertEqual(bridge.validate_settings({})['memoryPercent'], 75)
        self.assertEqual(bridge.validate_settings({})['cache'], 'int8_per_token_head')
        self.assertEqual(bridge.validate_settings({})['logMode'], 'memory')
        self.assertTrue(bridge.validate_settings({})['apiKeyEnabled'])
        self.assertEqual(bridge.validate_settings({'parallel': 32, 'cache': 'q8_0'})['parallel'], 32)

    def test_custom_context_auth_and_logging_validate_without_coercion(self):
        result = bridge.validate_settings({'context': 1537, 'apiKeyEnabled': False, 'logMode': 'off'})
        self.assertEqual(result['context'], 1537)
        self.assertFalse(result['apiKeyEnabled'])
        self.assertEqual(result['logMode'], 'off')
        for values in ({'apiKeyEnabled': 'false'}, {'logMode': 'none'}, {'context': None}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                bridge.validate_settings(values)

    def test_redeploy_credits_only_owned_process_on_matching_gpu(self):
        gpus = [{'uuid': 'GPU-a', 'total_memory_mib': 24000, 'free_memory_mib': 8000},
                {'uuid': 'GPU-b', 'total_memory_mib': 16000, 'free_memory_mib': 12000}]
        state = {'phase': 'ready', 'pid': 123, 'loaded_memory_mib': 11000,
                 'plan': {'gpu': {'uuid': 'GPU-a'}, 'budget': {'estimated_total_mib': 15000}}}
        output = type('Output', (), {'stdout': '999, GPU-a, 4000\n123, GPU-a, 10000\n123, GPU-b, 2000\n'})()
        with patch.object(bridge.subprocess, 'run', return_value=output):
            result = bridge.reclaim_memory(gpus, state)
        self.assertEqual(result[0]['redeploy_available_mib'], 18000)
        self.assertEqual(result[0]['reclaimable_memory_mib'], 10000)
        self.assertFalse(result[0]['reclaim_is_estimate'])
        self.assertEqual(result[1]['redeploy_available_mib'], 12000)
        self.assertEqual(gpus[0]['free_memory_mib'], 8000)

    def test_wddm_fallback_capped_and_stopped_service_not_credited(self):
        gpu = {'uuid': 'GPU-a', 'total_memory_mib': 24000, 'free_memory_mib': 8000}
        state = {'phase': 'ready', 'pid': 123, 'loaded_memory_mib': 30000,
                 'plan': {'gpu': {'uuid': 'GPU-a'}}}
        output = type('Output', (), {'stdout': '123, GPU-a, [N/A]\n'})()
        with patch.object(bridge.subprocess, 'run', return_value=output):
            result = bridge.reclaim_memory([gpu], state)[0]
            self.assertEqual(result['redeploy_available_mib'], 24000)
            self.assertTrue(result['reclaim_is_estimate'])
            state['phase'] = 'error'
            self.assertEqual(bridge.reclaim_memory([gpu], state)[0]['reclaimable_memory_mib'], 0)


    def test_error_hints_cover_oom_download_and_corruption(self):
        for message, expected in [('CUDA out of memory', 'memory'), ('Model SHA256 mismatch', 'integrity'),
                                  ('urlopen timed out', 'network'), ('nvidia-smi failed', 'gpu')]:
            self.assertEqual(bridge.friendly_error(RuntimeError(message))['code'], expected)




    def test_import_rejects_invalid_model_without_replacing_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = bridge.Bridge({'root': str(ROOT), 'data': directory, 'output': str(Path(directory)/'result.json'), 'args': {'profile': 'fast', 'path': str(Path(directory)/'source.gguf')}})
            instance.manifest['files']['fast'] = {'filename': 'test.gguf', 'size_bytes': 4, 'sha256': '0'*64}
            _, target = instance.model_item()
            target.write_bytes(b'good')
            (Path(directory)/'source.gguf').write_bytes(b'bad!')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                instance.import_model()
            self.assertEqual(target.read_bytes(), b'good')
            self.assertFalse(target.with_name('test.gguf.import').exists())

    def test_eof_handshake_cannot_start_service(self):
        with patch.object(sys, 'stdin', io.StringIO('')), patch.object(bridge, 'Bridge') as constructor:
            self.assertEqual(bridge.main(), 0)
            constructor.assert_not_called()


class SharedModelTests(unittest.TestCase):
    def make_bridge(self, base, version='new', content=b'current model', roots=None):
        instance = bridge.Bridge({'root': str(ROOT), 'data': str(base/version/'data'),
                                 'model_dir': str(base/'shared/models'),
                                 'model_search_roots': [str(p) for p in (roots or [base/version])],
                                 'output': str(base/'result.json'), 'args': {'profile': 'fast'}})
        instance.manifest['files'] = {'fast': {'filename': 'test.gguf', 'size_bytes': len(content),
                                             'sha256': hashlib.sha256(content).hexdigest()}}
        instance.manifest.update(checkpoint_dir='test-checkpoint',checkpoint_size_bytes=len(content),
            checkpoint_files={'model.safetensors':{'size_bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}},
            checkpoints={},repositories={})
        return instance

    def test_raw_checkpoint_reuse_and_removal_preserve_original(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve()
            source=base/'new/models/test-checkpoint/model.safetensors'
            source.parent.mkdir(parents=True);source.write_bytes(b'current model')
            instance=self.make_bridge(base)
            instance.reuse_models()
            bundle=instance.model_item()[1]
            checkpoint=bundle.parent/'test-checkpoint'
            self.assertFalse(bundle.exists())
            self.assertEqual((checkpoint/'model.safetensors').read_bytes(),b'current model')
            with patch.object(bridge,'detect_gpus',return_value=[]):
                self.assertTrue(instance.inventory()['models'][0]['installed'])
            with patch.object(bridge,'download_checkpoint') as download:
                self.assertEqual(instance.download()['result'],'already_verified')
                download.assert_not_called()
            instance.remove_model();instance.reuse_models()
            self.assertFalse(checkpoint.exists())
            self.assertEqual(source.read_bytes(),b'current model')

    def test_portable_versions_and_installed_app_reuse_same_verified_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            old = base/'HyMT-0.1.2-windows-x64-portable/data/models/test.gguf'
            old.parent.mkdir(parents=True)
            old.write_bytes(b'current model')
            first = self.make_bridge(base, 'HyMT-0.1.3-windows-x64-portable')
            first.reuse_models()
            item, shared = first.model_item()
            self.assertEqual(shared.read_bytes(), old.read_bytes())
            self.assertTrue(old.exists())
            second = self.make_bridge(base, 'installed')
            self.assertEqual(second.model_item()[1], shared)
            with patch.object(bridge, 'detect_gpus', return_value=[]):
                inventory = second.inventory()
            self.assertTrue(inventory['models'][0]['installed'])
            self.assertEqual(inventory['model_dir'], str(base/'shared/models'))
            with patch.object(bridge, 'install_model') as download:
                second.download()
                download.assert_not_called()
            second.verify_model(item, shared)
            # Full verification rejects same-size corruption even after discovery.
            shared.write_bytes(b'corrupt model')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                second.verify_model(item, shared)

    def test_wrong_same_size_legacy_file_is_skipped_for_valid_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            for name, content in [('bad', b'incorrect!!!!'), ('good', b'current model')]:
                path = base/name/'models/test.gguf'
                path.parent.mkdir(parents=True)
                path.write_bytes(content)
            instance = self.make_bridge(base, roots=[base/'bad', base/'good'])
            instance.reuse_models()
            self.assertEqual(instance.model_item()[1].read_bytes(), b'current model')
            self.assertTrue(instance.model_warnings)

    def test_cross_volume_copy_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source = base/'new/data/models/test.gguf'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'current model')
            instance = self.make_bridge(base)
            with patch.object(bridge.os, 'link', side_effect=OSError('cross volume')):
                instance.reuse_models()
            self.assertEqual(instance.model_item()[1].read_bytes(), source.read_bytes())
            self.assertFalse(instance.model_item()[1].with_name('test.gguf.reuse').exists())

    def test_revisions_with_same_filename_coexist(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            first = self.make_bridge(base, content=b'version one')
            second = self.make_bridge(base, content=b'version two')
            old = first.model_item()[1]
            new = second.model_item()[1]
            self.assertNotEqual(old, new)
            old.write_bytes(b'version one')
            new.write_bytes(b'version two')
            second.remove_model()
            self.assertEqual(old.read_bytes(), b'version one')

    def test_deleted_shared_model_is_not_reimported_from_legacy_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source = base/'new/data/models/test.gguf'
            source.parent.mkdir(parents=True)
            source.write_bytes(b'current model')
            instance = self.make_bridge(base)
            instance.reuse_models()
            instance.remove_model()
            instance.reuse_models()
            self.assertFalse(instance.model_item()[1].exists())
            self.assertTrue(source.exists())
            instance.args['path'] = str(source)
            instance.import_model()
            self.assertTrue(instance.model_item()[1].exists())
            self.assertFalse(instance.model_item()[1].with_suffix('.removed').exists())

    def test_portable_local_directory_reuses_registered_models_from_other_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            first = self.make_bridge(base, 'first')
            first.models = base/'portable-one/models'
            first.model_registry = base/'user/model-locations.json'
            item, original = first.model_item()
            original.write_bytes(b'current model')
            first.remember_model_dir()
            second = self.make_bridge(base, 'second')
            second.models = base/'different-drive/new-folder/models'
            second.model_registry = first.model_registry
            # Prove reuse comes from the shared path index, not nearby scanning.
            second.search_roots = []
            second.reuse_models()
            self.assertNotEqual(second.model_item()[1], original)
            self.assertEqual(second.model_item()[1].read_bytes(), b'current model')
            self.assertEqual(original.read_bytes(), b'current model')
            self.assertIn(first.models, second.registered_model_dirs())
            self.assertIn(second.models, second.registered_model_dirs())
            # Moving a portable folder preserves local weights without the index.
            second.model_registry = None
            second.verify_model(item, second.model_item()[1])

    def test_unwritable_registry_does_not_break_portable_local_import(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            instance = self.make_bridge(base)
            instance.model_registry = base/'blocked/locations.json'
            (base/'blocked').write_text('not a directory')
            source = base/'model.gguf'
            source.write_bytes(b'current model')
            instance.args['path'] = str(source)
            instance.import_model()
            self.assertEqual(instance.model_item()[1].read_bytes(), b'current model')


class PortableUpdateTests(unittest.TestCase):
    def archive(self, extra):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            archive.writestr('hymt-desktop.exe', b'MZ')
            archive.writestr('payload/scripts/desktop_bridge.py', '# worker')
            for name in extra:
                info = zipfile.ZipInfo('safe')
                info.filename = name
                archive.writestr(info, 'bad')
        buffer.seek(0)
        return zipfile.ZipFile(buffer)

    def test_rejects_traversal_data_overwrite_and_case_collision(self):
        for name in ('../escape', '/absolute', 'payload/../escape', 'payload/a:stream',
                     'data/models/existing.gguf', 'payload\\escape', 'HYMT-DESKTOP.EXE', 'payload/file. ',
                     'vcruntime140.dll/child', 'unlisted.dll'):
            with self.subTest(name=name), self.archive([name]) as archive, self.assertRaises(ValueError):
                updater.validate_entries(archive)

    def test_accepts_complete_portable_layout(self):
        with self.archive(['portable.json', 'payload/runtime/vllm/python.exe', *updater.ROOT_DLLS]) as archive:
            self.assertGreater(updater.validate_entries(archive), 0)

    def test_runtime_dll_update_rolls_back_even_when_one_dll_was_new(self):
        with tempfile.TemporaryDirectory() as directory:
            install=Path(directory).resolve();(install/'portable.json').write_text('{}')
            (install/updater.EXE).write_bytes(b'old');(install/'payload').mkdir()
            staging=install/'data/updates';(staging/'next/payload').mkdir(parents=True)
            (staging/'next'/updater.EXE).write_bytes(b'new')
            for name in updater.ROOT_DLLS: (staging/'next'/name).write_bytes(b'new dll')
            (install/updater.ROOT_DLLS[1]).write_bytes(b'old dll')
            rename=updater.rename_retry
            def fail(source,target):
                if source==staging/'next'/updater.ROOT_DLLS[1]: raise OSError('locked DLL')
                rename(source,target)
            with patch.object(updater,'wait_parent'),patch.object(updater,'rename_retry',side_effect=fail),self.assertRaises(OSError):
                updater.apply(install,staging,123)
            self.assertEqual((install/updater.EXE).read_bytes(),b'old')
            self.assertFalse((install/updater.ROOT_DLLS[0]).exists())
            self.assertEqual((install/updater.ROOT_DLLS[1]).read_bytes(),b'old dll')

    def test_cleanup_never_escapes_staging_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)/'stage'
            parent.mkdir()
            outside = Path(directory)/'keep'
            outside.mkdir()
            with self.assertRaises(ValueError):
                updater.confined_remove(outside, parent)
            self.assertTrue(outside.exists())

    def test_failed_replace_rolls_back_and_preserves_user_data(self):
        with tempfile.TemporaryDirectory() as directory:
            install = Path(directory).resolve()
            (install/'portable.json').write_text('{}')
            (install/updater.EXE).write_bytes(b'old exe')
            (install/'payload').mkdir()
            (install/'payload/marker').write_text('old runtime')
            staging = install/'data/updates'
            (staging/'next/payload').mkdir(parents=True)
            (staging/'next'/updater.EXE).write_bytes(b'new exe')
            (staging/'next/payload/marker').write_text('new runtime')
            (install/'data/model').write_text('user weights')
            rename = updater.rename_retry
            def fail_once(source, target):
                if source == staging/'next/payload':
                    raise OSError('locked payload')
                rename(source, target)
            with patch.object(updater, 'wait_parent'), patch.object(updater, 'rename_retry', side_effect=fail_once), self.assertRaisesRegex(OSError, 'locked payload'):
                updater.apply(install, staging, 123)
            self.assertEqual((install/updater.EXE).read_bytes(), b'old exe')
            self.assertEqual((install/'payload/marker').read_text(), 'old runtime')
            self.assertEqual((install/'data/model').read_text(), 'user weights')


if __name__ == '__main__':
    unittest.main()
