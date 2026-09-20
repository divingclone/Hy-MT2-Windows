"""Offline regression tests: no model, network, CUDA, pip, or external account."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import setup_model as setup

PAYLOAD = b"verified GGUF fixture\x00" * 100
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://huggingface.co/owner/repo/resolve/main/model.gguf"


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers if headers is not None else {"Content-Length": str(len(payload))}

    def geturl(self):
        return "https://cdn.example/model.gguf"


class Opener:
    def __init__(self, callback):
        self.callback = callback
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        return self.callback(request)


class SetupModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hymt-setup-test-")
        self.root = Path(self.tmp.name)
        self.destination = self.root / "model.gguf"
        self.addCleanup(self.tmp.cleanup)
        self.quiet = patch("sys.stdout", new_callable=io.StringIO)
        self.quiet.start()
        self.addCleanup(self.quiet.stop)

    def install(self, opener, **kwargs):
        return setup.install_model(URL, self.destination, len(PAYLOAD), DIGEST, opener=opener, retries=0, **kwargs)

    def test_success_installs_only_verified_file_and_reuses_it(self):
        opener = Opener(lambda _: Response(PAYLOAD))
        self.assertEqual(self.install(opener), "downloaded_verified")
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertFalse(self.destination.with_name("model.gguf.part").exists())
        self.assertEqual(self.install(opener), "already_verified")
        self.assertEqual(len(opener.requests), 1)

    def test_bad_sha_does_not_replace_existing_model(self):
        self.destination.write_bytes(b"keep the old model")
        opener = Opener(lambda _: Response(b"x" * len(PAYLOAD)))
        with self.assertRaisesRegex(setup.SetupError, "SHA256"):
            self.install(opener)
        self.assertEqual(self.destination.read_bytes(), b"keep the old model")

    def test_partial_download_resumes_at_exact_offset(self):
        partial = self.destination.with_name("model.gguf.part")
        partial.write_bytes(PAYLOAD[:123])

        def response(request):
            self.assertEqual(request.get_header("Range"), "bytes=123-")
            return Response(PAYLOAD[123:], 206, {"Content-Range": f"bytes 123-{len(PAYLOAD)-1}/{len(PAYLOAD)}",
                                               "Content-Length": str(len(PAYLOAD)-123)})
        self.install(Opener(response))
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_range_ignored_restarts_instead_of_appending(self):
        self.destination.with_name("model.gguf.part").write_bytes(b"stale")
        self.install(Opener(lambda _: Response(PAYLOAD)))
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_wrong_range_or_size_never_publishes(self):
        cases = [Response(PAYLOAD, 206, {"Content-Range": f"bytes 1-{len(PAYLOAD)}/{len(PAYLOAD)}"}),
                 Response(PAYLOAD, 200, {"Content-Length": "999"})]
        for response in cases:
            with self.subTest(response=response), self.assertRaises(setup.SetupError):
                self.install(Opener(lambda _, result=response: result))
            self.assertFalse(self.destination.exists())

    def test_short_transfer_retains_partial_for_retry(self):
        with self.assertRaisesRegex(setup.SetupError, "interrupted"):
            self.install(Opener(lambda _: Response(PAYLOAD[:123], 200, {})))
        self.assertFalse(self.destination.exists())

    def test_complete_partial_is_verified_without_network(self):
        self.destination.with_name("model.gguf.part").write_bytes(PAYLOAD)
        opener = Opener(lambda _: self.fail("download was unnecessary"))
        self.assertEqual(self.install(opener), "resumed_verified")

    def test_manifest_rejects_path_traversal_and_bad_sha(self):
        path = self.root / "manifest.json"
        item = {"filename": "model.gguf", "size_bytes": len(PAYLOAD), "sha256": DIGEST}
        manifest = {"schema_version": 1, "files": {"official": item, "fast": dict(item, filename="fast.gguf")}}
        path.write_text(json.dumps(manifest), encoding="utf-8")
        setup.load_manifest(path)
        for filename, digest in (("../model.gguf", DIGEST), ("model.gguf", "bad")):
            manifest["files"]["official"] = dict(item, filename=filename, sha256=digest)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(setup.SetupError):
                setup.load_manifest(path)

    def test_auto_selection_uses_build_metadata_without_model_files(self):
        gpu = {"index": 0, "uuid": "GPU-test", "name": "fixture", "free_memory_mib": 32000,
               "compute_capability": "12.0"}
        with patch.object(setup, "detect_gpus", return_value=[gpu]), patch.object(setup, "choose_binary") as binary:
            binary.return_value = (self.root, {"nvfp4_compute_capabilities": ["12.0"]}, {})
            self.assertEqual(setup.select_profile(self.root, "auto")[0], "fast")
            binary.return_value = (self.root, {"nvfp4_compute_capabilities": []}, {})
            self.assertEqual(setup.select_profile(self.root, "auto")[0], "official")

    def test_explicit_profile_does_not_probe_gpu(self):
        with patch.object(setup, "detect_gpus", side_effect=AssertionError("GPU probe")):
            self.assertEqual(setup.select_profile(self.root, "official")[0], "official")

    def test_redirect_strips_token_for_cdn_and_rejects_http(self):
        handler = setup.HTTPSRedirectHandler()
        request = urllib.request.Request(URL, headers={"Authorization": "Bearer unit-test-placeholder"})
        redirected = handler.redirect_request(request, None, 302, "Found", {}, "https://cdn.example/model.gguf")
        self.assertIsNone(redirected.get_header("Authorization"))
        with self.assertRaises(setup.SetupError):
            handler.redirect_request(request, None, 302, "Found", {}, "http://cdn.example/model.gguf")

    def test_repo_override_cannot_turn_into_a_url_or_parent_path(self):
        self.assertEqual(setup.model_url("owner/repo", "main", "model.gguf"), URL)
        for repo in (None, "https://example.com", "../repo", "owner/repo/extra"):
            with self.assertRaises(setup.SetupError):
                setup.model_url(repo, "main", "model.gguf")

    def test_invalid_token_is_rejected_without_echoing_it(self):
        opener = Opener(lambda _: self.fail("invalid token should not be sent"))
        with self.assertRaisesRegex(setup.SetupError, "invalid characters") as caught:
            self.install(opener, token="unit-test\nplaceholder")
        self.assertNotIn("placeholder", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
