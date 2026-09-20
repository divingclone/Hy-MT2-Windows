import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_webview_runtime as browser


class Response(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content);self.status=status;self.headers=headers or {}


class WebViewTests(unittest.TestCase):
    def test_resume_and_server_ignoring_range(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(browser,'SIZE',6):
            cab=Path(temp)/'download.part';cab.write_bytes(b'abc')
            with patch.object(browser.urllib.request,'urlopen',return_value=Response(b'def',206,{'Content-Range':'bytes 3-5/6'})) as request:
                browser.download(cab,lambda _:None)
                self.assertEqual(request.call_args.args[0].get_header('Range'),'bytes=3-')
            self.assertEqual(cab.read_bytes(),b'abcdef')
            cab.write_bytes(b'abc')
            with patch.object(browser.urllib.request,'urlopen',return_value=Response(b'abcdef')):
                browser.download(cab,lambda _:None)
            self.assertEqual(cab.read_bytes(),b'abcdef')

    def test_bad_range_preserves_partial(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(browser,'SIZE',6):
            cab=Path(temp)/'download.part';cab.write_bytes(b'abc')
            with patch.object(browser.urllib.request,'urlopen',return_value=Response(b'bad',206,{'Content-Range':'bytes 0-2/6'})):
                with self.assertRaises(ValueError):browser.download(cab,lambda _:None)
            self.assertEqual(cab.read_bytes(),b'abc')

    def test_bad_hash_never_extracts_and_does_not_poison_retry(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(browser,'SIZE',3):
            target=Path(temp)/browser.VERSION
            cab=target.with_name(target.name+'.cab.part');cab.write_bytes(b'bad')
            with patch.object(browser.subprocess,'run') as extract:
                with self.assertRaisesRegex(ValueError,'digest mismatch'):browser.ensure_runtime(target)
                extract.assert_not_called()
            self.assertFalse(target.exists());self.assertFalse(cab.exists())

    def test_publish_after_signature_and_reuse_without_network(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(browser,'SHA256',hashlib.sha256(b'cab').hexdigest()):
            cab=Path(temp)/'test.cab';cab.write_bytes(b'cab');target=Path(temp)/'runtime'
            def extract(command,**kwargs):
                source=Path(command[-1])/f'Microsoft.WebView2.FixedVersionRuntime.{browser.VERSION}.x64'
                source.mkdir()
                (source/'msedgewebview2.exe').write_bytes(b'exe');(source/'msedge.dll').write_bytes(b'dll')
            with patch.object(browser.subprocess,'run',side_effect=extract),patch.object(browser.subprocess,'check_output',return_value='{"status":"Invalid","signer":"CN=Microsoft Corporation,"}'):
                with self.assertRaisesRegex(ValueError,'signature'):browser.ensure_runtime(target,cab)
                self.assertFalse(target.exists())
            with patch.object(browser.subprocess,'run',side_effect=extract),patch.object(browser.subprocess,'check_output',return_value='{"status":"Valid","signer":"CN=Microsoft Corporation,"}'):
                browser.ensure_runtime(target,cab)
                self.assertTrue(browser.ready(target))
            stale=target.parent/f'webview-build-{browser.VERSION}-interrupted'
            stale.mkdir();(stale/'unfinished.dll').write_bytes(b'partial')
            unrelated=target.parent/'keep.txt';unrelated.write_bytes(b'keep')
            with patch.object(browser,'download',side_effect=AssertionError('Unexpected download')):
                self.assertEqual(browser.ensure_runtime(target),target.resolve())
            self.assertFalse(stale.exists());self.assertEqual(unrelated.read_bytes(),b'keep')


if __name__=='__main__':unittest.main()
