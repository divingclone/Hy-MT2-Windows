"""File safety and process ownership checks; no model or GPU required."""
import os
import contextlib
import io
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import serve
import gpu_config
import run_native_experiment
import translate_batch
from translate_batch import check_distinct_paths


class LauncherSafetyTests(unittest.TestCase):
    def test_source_log_collision_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'sample.log'
            source.write_bytes(b'original request')
            with self.assertRaises(ValueError):
                check_distinct_paths({'input': source, 'output': source.with_suffix('.jsonl'), 'log': source})
            self.assertEqual(source.read_bytes(), b'original request')

    def test_output_sidecar_collision(self):
        path = Path(tempfile.gettempdir())/'translation.log'
        with self.assertRaises(ValueError):
            check_distinct_paths({'output': path, 'log': path})

    def test_hardlink_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            source, alias = Path(directory)/'input.jsonl', Path(directory)/'output.jsonl'
            source.write_bytes(b'original')
            os.link(source, alias)
            with self.assertRaises(ValueError):
                check_distinct_paths({'input': source, 'output': alias})

    def test_distinct_paths(self):
        root = Path(tempfile.gettempdir())
        check_distinct_paths({'input': root/'input.jsonl', 'output': root/'output.jsonl', 'log': root/'output.log'})

    @unittest.skipUnless(os.name == 'nt', 'Windows process management')
    def test_live_label_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(serve, 'ROOT', Path(directory)):
            results = Path(directory)/'results'
            results.mkdir()
            pid_file = results/'server.pid'
            contents = str(os.getpid())
            pid_file.write_text(contents, encoding='ascii')
            with self.assertRaisesRegex(RuntimeError, 'already manages a live process'):
                serve.check_managed_label('server')
            self.assertEqual(pid_file.read_text(encoding='ascii'), contents)

    def test_no_existing_label(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(serve, 'ROOT', Path(directory)):
            serve.check_managed_label('server')

    @unittest.skipUnless(os.name == 'nt' and shutil.which('powershell.exe'), 'Windows process management')
    def test_process_creation_time_matches_windows_process_metadata(self):
        with subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                              creationflags=subprocess.CREATE_NO_WINDOW) as process:
            created = serve.process_creation_filetime(process)
            inspected = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                                        f'(Get-Process -Id {process.pid}).StartTime.ToUniversalTime().ToFileTimeUtc()'],
                                       capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        self.assertEqual(inspected.returncode, 0, inspected.stderr)
        self.assertEqual(created, inspected.stdout.strip())


@unittest.skipUnless(os.name == 'nt' and shutil.which('powershell.exe'), 'PowerShell process management')
class StopServerTests(unittest.TestCase):
    def test_stopper_checks_custom_executable_and_process_ownership(self):
        original = Path(__file__).resolve().parents[1]/'scripts/stop-server.ps1'
        for scenario in ('custom', 'old-config', 'wrong-path', 'wrong-name', 'wrong-pid', 'reused-pid', 'missing-config', 'relative-path'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'scripts').mkdir()
                (root/'results').mkdir()
                shutil.copyfile(original, root/'scripts/stop-server.ps1')
                executable = root/'build/custom/bin/llama-server.exe'
                config = {'mode': 'server', 'label': 'fixture', 'executable': str(executable),
                          'command': [str(executable), '--model', 'fixture.gguf'], 'managed_pid': 4321,
                          'process_creation_filetime': '133801632000000000'}
                running = str(executable)
                if scenario == 'old-config':
                    config.pop('managed_pid')
                    config.pop('process_creation_filetime')
                elif scenario == 'wrong-path':
                    running = str(root/'other/llama-server.exe')
                elif scenario == 'wrong-name':
                    running = config['executable'] = str(root/'python.exe')
                    config['command'][0] = running
                elif scenario == 'relative-path':
                    config['executable'] = 'build/custom/bin/llama-server.exe'
                    config['command'][0] = config['executable']
                elif scenario == 'wrong-pid':
                    config['managed_pid'] = 1234
                elif scenario == 'reused-pid':
                    config['process_creation_filetime'] = '133801632000000001'
                pid_file = root/'results/fixture.pid'
                pid_file.write_text('4321', encoding='ascii')
                if scenario != 'missing-config':
                    (root/'results/fixture.config.json').write_text(json.dumps(config), encoding='utf-8')
                (root/'driver.ps1').write_text('''$ErrorActionPreference = 'Stop'
$global:hyMockProcess = [pscustomobject]@{ Id = 4321; Path = $env:HY_TEST_RUNNING_EXE; StartTime = [DateTime]::FromFileTimeUtc(133801632000000000) }
function Get-Process { [CmdletBinding()] param([int]$Id) return $global:hyMockProcess }
function Stop-Process { [CmdletBinding()] param($InputObject) Set-Content -LiteralPath $env:HY_TEST_STOP_MARKER -Value $InputObject.Id -Encoding ASCII }
& "$PSScriptRoot/scripts/stop-server.ps1" -Label fixture
''', encoding='utf-8')
                marker = root/'stopped.txt'
                environment = dict(os.environ, HY_TEST_RUNNING_EXE=running, HY_TEST_STOP_MARKER=str(marker))
                result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(root/'driver.ps1')],
                                        env=environment, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15,
                                        creationflags=subprocess.CREATE_NO_WINDOW)
                if scenario in ('custom', 'old-config'):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(marker.read_text(encoding='ascii').strip(), '4321')
                    self.assertFalse(pid_file.exists())
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('refusing', result.stderr)
                    self.assertFalse(marker.exists())
                    self.assertEqual(pid_file.read_text(encoding='ascii'), '4321')


class CacheLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root/'candidate.gguf'
        self.model.write_bytes(b'mock model')
        self.binary_dir = self.root/'bin'
        self.binary_dir.mkdir()
        for name in ('hy-batch.exe', 'llama-server.exe'):
            (self.binary_dir/name).write_bytes(b'never executed')
        self.source = self.root/'input.jsonl'
        self.source.write_text('{"id":"a","text":"Hello","target_lang":"Chinese"}\n', encoding='utf-8')

    def configuration(self, key='f16', value='f16'):
        return dict(gpu_config.memory_plan(mode='batch', free_mib=4096, model_bytes=1024,
                                           parallel=1, cache_type_k=key, cache_type_v=value),
                    model=str(self.model), binary_dir=str(self.binary_dir), cuda_visible_devices='GPU-test',
                    profile='fast', warnings=[])

    def test_batch_model_override_and_cache_flags_reach_native_command(self):
        for cache_type in ('f16', 'q8_0'):
            destination = self.root/f'{cache_type}.jsonl'
            argv = ['translate_batch.py', str(self.source), str(destination), '--model', str(self.model),
                    '--cache-type-k', cache_type, '--cache-type-v', cache_type]
            with patch.object(sys, 'argv', argv), patch.object(translate_batch, 'ROOT', self.root), \
                    patch.object(translate_batch, 'resolve_config', return_value=self.configuration(cache_type, cache_type)) as resolve, \
                    patch.object(translate_batch, 'ensure_cache_type_support') as support, \
                    patch.object(translate_batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(translate_batch.main(), 0)
            self.assertEqual(resolve.call_args.kwargs['model_override'], self.model)
            self.assertEqual(resolve.call_args.kwargs['cache_type_k'], cache_type)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('-m') + 1], str(self.model))
            self.assertEqual(command[command.index('--cache-type-k') + 1], cache_type)
            self.assertEqual(command[command.index('--cache-type-v') + 1], cache_type)
            support.assert_called_once()
            saved = json.loads(destination.with_suffix('.config.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['command'], command)
            self.assertEqual(saved['cache_type_v'], cache_type)

    def test_unsupported_batch_cache_does_not_create_output_artifacts(self):
        destination = self.root/'new-output'/'translations.jsonl'
        with patch.object(sys, 'argv', ['translate_batch.py', str(self.source), str(destination), '--cache-type-k', 'q8_0']), \
                patch.object(translate_batch, 'ROOT', self.root), \
                patch.object(translate_batch, 'resolve_config', return_value=self.configuration('q8_0', 'f16')), \
                patch.object(translate_batch, 'ensure_cache_type_support', side_effect=gpu_config.ConfigError('rebuild')), \
                patch.object(translate_batch.subprocess, 'run') as run:
            with self.assertRaisesRegex(gpu_config.ConfigError, 'rebuild'):
                translate_batch.main()
        self.assertFalse(destination.parent.exists())
        run.assert_not_called()

    def test_batch_output_cannot_overwrite_selected_model(self):
        with patch.object(sys, 'argv', ['translate_batch.py', str(self.source), str(self.model), '--model', str(self.model)]), \
                patch.object(translate_batch, 'ROOT', self.root), \
                patch.object(translate_batch, 'resolve_config', return_value=self.configuration()), \
                patch.object(translate_batch.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'model'):
                translate_batch.main()
        self.assertEqual(self.model.read_bytes(), b'mock model')
        run.assert_not_called()

    def test_server_passes_cache_types_with_flash_attention(self):
        argv = ['serve.py', '--cache-type-k', 'q8_0', '--cache-type-v', 'q4_0', '--model', str(self.model),
                '--binary-dir', str(self.binary_dir)]
        with patch.object(sys, 'argv', argv), patch.object(serve, 'ROOT', self.root), \
                patch.object(serve, 'resolve_config', return_value=self.configuration('q8_0', 'q4_0')) as resolve, \
                patch.object(serve, 'ensure_cache_type_support') as support, \
                patch.object(serve.socket, 'socket'), patch.object(serve.subprocess, 'call', return_value=0) as run:
            self.assertEqual(serve.main(), 0)
        self.assertEqual(resolve.call_args.kwargs['cache_type_v'], 'q4_0')
        self.assertEqual(resolve.call_args.kwargs['binary_dir'], self.binary_dir)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--cache-type-k') + 1], 'q8_0')
        self.assertEqual(command[command.index('--cache-type-v') + 1], 'q4_0')
        self.assertEqual(command[command.index('-fa') + 1], 'on')
        self.assertIn('--no-context-shift', command)
        support.assert_called_once()

    def test_server_rejects_removed_variant_selection(self):
        argv = ['serve.py', '--binary-dir', str(self.binary_dir), '--variant', 'optimized']
        with patch.object(sys, 'argv', argv), patch.object(serve, 'resolve_config') as resolve, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as failure:
                serve.main()
        self.assertEqual(failure.exception.code, 2)
        resolve.assert_not_called()

    def test_background_server_records_pid_and_creation_time(self):
        argv = ['serve.py', '--binary-dir', str(self.binary_dir), '--background']
        process = MagicMock(pid=4321)
        process.poll.return_value = None
        http = MagicMock()
        http.open.return_value.__enter__.return_value = io.StringIO('{"status":"ok"}')
        with patch.object(sys, 'argv', argv), patch.object(serve, 'ROOT', self.root), \
                patch.object(serve, 'resolve_config', return_value=self.configuration()), \
                patch.object(serve, 'ensure_cache_type_support'), patch.object(serve.socket, 'socket'), \
                patch.object(serve.subprocess, 'Popen', return_value=process), \
                patch.object(serve, 'process_creation_filetime', return_value='133801632000000000'), \
                patch.object(serve.urllib.request, 'build_opener', return_value=http), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(serve.main(), 0)
        saved = json.loads((self.root/'results/server.config.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['managed_pid'], 4321)
        self.assertEqual(saved['process_creation_filetime'], '133801632000000000')
        self.assertEqual(Path(saved['executable']).resolve(), (self.binary_dir/'llama-server.exe').resolve())
        self.assertEqual(saved['command'][0], saved['executable'])
        self.assertEqual((self.root/'results/server.pid').read_text(encoding='ascii'), '4321')

    def test_experiment_dry_run_records_cache_without_executing_binary(self):
        for cache_type in ('f16', 'q8_0'):
            output = io.StringIO()
            argv = ['run_native_experiment.py', '--label', 'cache-test', '--dry-run', '--cases', '32', '--rounds', '1',
                    '--cache-type-k', cache_type, '--cache-type-v', cache_type]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(output), \
                    patch.object(run_native_experiment.subprocess, 'run') as run:
                self.assertEqual(run_native_experiment.run(run_native_experiment.parse_args()), 0)
            run.assert_not_called()
            saved = json.loads(output.getvalue())
            self.assertEqual(saved['cache_type_k'], cache_type)
            self.assertFalse(saved['cache_cli_support_checked'])
            self.assertIn('--cache-type-k', saved['native_command'])
            self.assertEqual(saved['native_command'][saved['native_command'].index('--cache-type-k') + 1], cache_type)


if __name__ == '__main__':
    unittest.main()
