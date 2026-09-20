// Optional CUDA libraries are loaded only from beside torch_cuda.dll.
// Missing features fail as a normal PyTorch RuntimeError, never a fake result.
#include <windows.h>
#include <delayimp.h>
#include <c10/util/Exception.h>
#include <cstring>
#include <cwchar>

namespace {
bool optional_library(const char* name) {
  return std::strcmp(name, "cufft64_12.dll") == 0 ||
      std::strcmp(name, "cusparse64_12.dll") == 0 ||
      std::strcmp(name, "cusolver64_12.dll") == 0;
}

FARPROC WINAPI load_optional_cuda(unsigned notification, PDelayLoadInfo info) {
  if (!optional_library(info->szDll)) {
    return nullptr;
  }
  if (notification == dliNotePreLoadLibrary) {
    HMODULE self = nullptr;
    TORCH_CHECK(GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&load_optional_cuda), &self),
        "HyMT: cannot locate the CUDA backend module");
    wchar_t path[32768];
    const DWORD length = GetModuleFileNameW(self, path, 32768);
    TORCH_CHECK(length && length < 32768, "HyMT: CUDA backend path is too long");
    wchar_t* filename = std::wcsrchr(path, L'\\');
    TORCH_CHECK(filename, "HyMT: CUDA backend path has no directory");
    ++filename;
    const size_t available = 32768 - (filename - path);
    const int copied = MultiByteToWideChar(CP_UTF8, 0, info->szDll, -1,
                                         filename, static_cast<int>(available));
    TORCH_CHECK(copied, "HyMT: invalid optional CUDA library name");
    const HMODULE library = LoadLibraryExW(path, nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    TORCH_CHECK(library, "HyMT's compact PyTorch build does not include ", info->szDll,
        ". This operation needs the full CUDA FFT/sparse/solver runtime. Windows error: ",
        GetLastError());
    return reinterpret_cast<FARPROC>(library);
  }
  if (notification == dliFailGetProc) {
    TORCH_CHECK(false, "HyMT: incompatible optional CUDA library ", info->szDll);
  }
  return nullptr;
}
} // namespace

extern "C" const PfnDliHook __pfnDliNotifyHook2 = load_optional_cuda;
extern "C" const PfnDliHook __pfnDliFailureHook2 = load_optional_cuda;
