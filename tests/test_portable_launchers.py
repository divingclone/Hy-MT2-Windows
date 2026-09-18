"""File safety and process ownership checks; no model or GPU required."""
import os
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import serve
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


if __name__ == '__main__':
    unittest.main()
