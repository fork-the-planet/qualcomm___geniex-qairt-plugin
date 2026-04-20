#include "runtime.h"

#ifdef _WIN32
#  include <windows.h>
extern "C" IMAGE_DOS_HEADER __ImageBase;
#elif defined(__ANDROID__) || defined(__linux__)
#  include <dlfcn.h>
#endif

namespace geniex {

// Defined in a .cpp compiled exclusively into geniex_core so that the
// address anchors (__ImageBase / dladdr) always resolve to the geniex_core
// shared library, never to a consuming executable or a different DLL.

std::filesystem::path geniex_core_dir() {
#ifdef _WIN32
    // __ImageBase is a linker-generated symbol that always refers to the base
    // of the current module (geniex_core.dll), regardless of inlining.
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(reinterpret_cast<HMODULE>(&__ImageBase), path, MAX_PATH);
    auto result = std::filesystem::path(path).parent_path();
    GENIEX_LOG_INFO("geniex_core_dir: resolved to {}", result.string());
    return result;
#elif defined(__ANDROID__) || defined(__linux__)
    // dladdr resolves the shared object containing this function's code.
    // Because this .cpp is compiled only into libgeniex_core.so, the
    // address always points into the correct library.
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&geniex_core_dir), &info) && info.dli_fname) {
        auto result = std::filesystem::canonical(info.dli_fname).parent_path();
        GENIEX_LOG_INFO("geniex_core_dir: resolved to {}", result.string());
        return result;
    }
    throw std::runtime_error("geniex: cannot determine geniex_core library directory");
#else
    return std::filesystem::current_path();
#endif
}

}  // namespace geniex
