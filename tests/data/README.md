# Test data

This directory holds small fixture files used by the CTest generation smoke
tests (see `../CMakeLists.txt`).

## Required files

| File             | Used by                            | Notes                               |
|------------------|------------------------------------|-------------------------------------|
| `test_image.jpg` | `qwen2_5_vl_7b_generate_smoke`     | Any small JPEG (≤ 1 MB). Not committed. |

Drop a JPEG here locally before running `ctest`, or override on the CMake
command line:

```pwsh
cmake -B build -A ARM64 `
  -DGENIEX_BUILD_EXAMPLES=ON -DGENIEX_BUILD_VLM=ON -DGENIEX_BUILD_TESTS=ON `
  -DGENIEX_TEST_IMAGE="C:\path\to\any.jpg"
```

These files are intentionally *not* checked in to avoid bloating the repo.
