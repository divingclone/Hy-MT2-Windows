# Third-party notices and redistribution audit

This is a component inventory and source record, not a blanket license grant or
a legal assurance about a distributor's circumstances. Preserve original notices
and comply with each applicable license. No NVIDIA or Microsoft development SDK
or graphics driver is included in the runtime package.

| Component | Included materials and basis |
| --- | --- |
| Tencent Hy-MT2-1.8B | `model-Hy-MT2.txt`, Apache 2.0; modifications in `MODEL_CHANGES.md`. Official source: https://huggingface.co/tencent/Hy-MT2-1.8B/blob/main/LICENSE.txt |
| Modified llama.cpp / ggml | `llama.cpp-LICENSE.txt`, MIT; original repository https://github.com/ggml-org/llama.cpp . This package contains local inference changes, not an unmodified upstream release. |
| cpp-httplib, nlohmann/json, SHA-256, xxHash, rotate-bits | Original license texts are copied from the corresponding vendored sources without editing. |
| stb_image, subprocess.h, SHA-1 | `stb-LICENSE.txt`, `subprocess-LICENSE.txt`, `sha1-NOTICE.txt` extracted verbatim from the vendored source's notice blocks; source paths recorded in `SOURCES.json`. |
| NVIDIA CUDA runtime, cuBLAS, cuBLASLt | `NVIDIA-CUDA-EULA.txt` copied from the installed CUDA 13.0 toolkit. Attachment A names the redistributable runtime/BLAS families, subject to the agreement's application distribution requirements: https://docs.nvidia.com/cuda/archive/13.0.0/eula/index.html |
| NVIDIA CCCL headers compiled into CUDA code | `cccl-LICENSE.txt` includes Apache 2.0, LLVM exceptions and component notices from NVIDIA's v3.0.0 source release: https://github.com/NVIDIA/cccl/tree/v3.0.0 |
| CPython 3.12.13, standalone build 20260728 | Original `runtime/python/LICENSE.txt`, plus `python-3.12-license.txt` with the complete Python license documentation body and additional component notices, extracted from the archived HTML. The latter is a documentation snapshot for the 3.12 series, not proof of every binary component version. Build distribution: https://github.com/astral-sh/python-build-standalone/releases/tag/20260728 |
| OpenSSL 3.5.7 | `openssl-LICENSE.txt`, Apache 2.0, from matching OpenSSL tag. Runtime version was read from the Python DLLs and `ssl.OPENSSL_VERSION`: https://github.com/openssl/openssl/tree/openssl-3.5.7 |
| libffi | `libffi-LICENSE.txt`, MIT, from the official v3.4.6 source. The bundled DLL does not advertise an exact version; this license text is retained as attribution, not a claim that the DLL's version was identified: https://github.com/libffi/libffi/tree/v3.4.6 |
| SQLite | `sqlite-PUBLIC-DOMAIN.txt`, the project's complete public-domain article extracted from the archived HTML. The bundled DLL reports version 3.53.1: https://www.sqlite.org/copyright.html |
| liblzma | `xz-COPYING.txt` and `xz-0BSD.txt` from XZ 5.8.1 document liblzma's 0BSD terms. The exact linked liblzma version was not exposed by Python; the source tag is an attribution source, not a binary version claim: https://github.com/tukaani-project/xz/tree/v5.8.1 |
| Microsoft Visual C++ Runtime | Unmodified runtime DLLs, the complete official end-user license extracted as `msvc-runtime-license.txt`, and the complete Visual Studio 2022 redistribution article extracted as `msvc-redistribution.txt`. Redistribution authority comes from an applicable licensed developer product and its REDIST terms, **not** from the end-user runtime license alone: https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution |

The Microsoft runtime end-user agreement itself restricts onward distribution;
do not treat its inclusion as a redistribution permission. The official Visual
Studio REDIST list permits specified unmodified runtime files to be distributed
with an application by appropriately licensed developers, subject to their
Visual Studio terms. Python's original license also contains conditions for its
Windows binaries and Microsoft code. This inventory does not determine the
publisher's license entitlement or replace those terms.

The Python payload excludes pip/site-packages, Tcl/Tk, developer import libraries
and test extensions. Additional standard-library dependencies covered by the
Python license documentation include zlib, bzip2 and Expat; separate upstream
notices are provided for liblzma and SQLite. We
retain the original Python license even when it contains notices for excluded
components. No claim is made that a notice's presence means that component ships.

HTML license articles are distributed as UTF-8 TXT with their full article text,
terms, copyright notices and links retained. Website navigation, scripts, visual
decoration and duplicate source formats are omitted. The Microsoft runtime's
download landing page and Word copy are replaced by its complete TXT license.
Python's original runtime license is retained alongside the additional notices.

Downloaded and extracted notice source URLs/paths, extraction descriptions,
original source hashes and current TXT hashes are in `SOURCES.json`. Entries in
`omitted_source_materials` describe historical inputs, not files in this package.
The package's manifest independently hashes the actual files
distributed. A package dependency check follows normal and delay-load PE imports
and explicitly includes the dynamically loaded ggml backends and server module;
it cannot prove the absence of every possible runtime `LoadLibrary` call.
