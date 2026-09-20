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
                executable = root/'build/custom/bin/python.exe'
                config = {'mode': 'server', 'label': 'fixture', 'backend':'vllm', 'executable': str(executable),
                          'command': [str(executable), '--model', 'fixture.gguf'], 'managed_pid': 4321,
                          'process_creation_filetime': '133801632000000000'}
                running = str(executable)
                if scenario == 'old-config':
                    config.pop('managed_pid')
                    config.pop('process_creation_filetime')
                elif scenario == 'wrong-path':
                    running = str(root/'other/python.exe')
                elif scenario == 'wrong-name':
                    running = config['executable'] = str(root/'wrong.exe')
                    config['command'][0] = running
                elif scenario == 'relative-path':
                    config['executable'] = 'build/custom/bin/python.exe'
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
function taskkill.exe { $global:LASTEXITCODE = 0; Set-Content -LiteralPath $env:HY_TEST_STOP_MARKER -Value 4321 -Encoding ASCII }
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



if __name__=='__main__': unittest.main()
