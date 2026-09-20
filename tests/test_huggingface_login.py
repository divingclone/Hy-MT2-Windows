"""Maintainer login must not publish models or expose provider error details."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]


class HuggingFaceLoginTests(unittest.TestCase):
    def run_login(self, hub):
        output = io.StringIO()
        # Synthetic fixture; never read real credentials or contact Hugging Face.
        token = 'hf_' + 'x' * 12
        with patch.dict(os.environ), patch.dict(sys.modules, {
            'huggingface_hub': hub, 'httpx': MagicMock(),
        }), patch.object(sys, 'argv', ['publish_vllm_models.py', '--login-stdin']), \
                patch.object(sys, 'stdin', io.StringIO(token + '\n')), \
                contextlib.redirect_stdout(output):
            module = runpy.run_path(str(ROOT / 'scripts/publish_vllm_models.py'))
            module['main']()
        return token, output.getvalue()

    def test_login_returns_without_publication(self):
        hub = MagicMock()
        hub.HfApi.return_value.whoami.return_value = {'name': 'fixture-user'}
        token, output = self.run_login(hub)
        hub.login.assert_called_once_with(token=token, add_to_git_credential=False)
        hub.get_token.assert_not_called()
        hub.CommitOperationAdd.assert_not_called()
        hub.HfApi.return_value.create_repo.assert_not_called()
        hub.HfApi.return_value.create_commit.assert_not_called()
        self.assertIn('fixture-user', output)
        self.assertNotIn(token, output)
        gui = (ROOT / 'scripts/huggingface-login.ps1').read_text(encoding='utf-8')
        self.assertIn('scripts/publish_vllm_models.py --login-stdin', gui)

    def test_provider_error_is_sanitized(self):
        hub = MagicMock()
        hub.HfApi.return_value.whoami.side_effect = RuntimeError('secret-provider-detail')
        with self.assertRaisesRegex(RuntimeError, '^Hugging Face login failed;') as caught:
            self.run_login(hub)
        self.assertNotIn('secret-provider-detail', str(caught.exception))
        hub.login.assert_not_called()
        hub.get_token.assert_not_called()
        hub.HfApi.return_value.create_commit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
