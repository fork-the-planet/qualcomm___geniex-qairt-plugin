# Tests

CTest-driven test suite for the `geniex` LLM and VLM pipelines. Tests exercise
the real `LLMPipeline` / `VLMPipeline` on-device and therefore **require
Snapdragon NPU hardware** — they cannot run on the GitHub-hosted CI runner.

## Layout

```
tests/
├── CMakeLists.txt   # Per-(modality,model,mode) add_test() registrations
├── llm.cpp          # → binary `llm_test`
├── vlm.cpp          # → binary `vlm_test`
└── README.md        # This file
```

One binary per modality. Each binary takes `--mode <name>` to select the test
type (currently only `generation` is implemented; the design accommodates
future `tokenizer`, `kvcache`, etc. modes without adding new binaries).

## Prerequisites

| Requirement | Notes |
|---|---|
| Snapdragon NPU host | QAIRT/HTP runtime must be able to load the model graphs. |
| Model files | Stage `.bin` shards + `tokenizer.json` + `htp_backend_ext_config.json` under `<repo>/modelfiles/<name>/` matching the layout the examples expect. The exact shard filenames are enumerated in the `modelFilesTable` at the top of `llm.cpp`. |
| Test image (VLM only) | A JPEG/PNG at `<repo>/modelfiles/assets/test_image.png` (already committed). Override at run time with `--image <path>`. |

## Build

```pwsh
cmake -B build -A ARM64 `
      -DGENIEX_BUILD_VLM=ON `
      -DGENIEX_BUILD_TESTS=ON
cmake --build build --config Release -j
```

Relevant CMake options:

| Option | Default | Purpose |
|---|---|---|
| `GENIEX_BUILD_TESTS` | `OFF` | Enable this whole subtree. |
| `GENIEX_BUILD_VLM` | `OFF` | Required for `vlm_test` to build (the VLM tier is optional). |

Defaults for prompt, `--max-tokens`, `--min-tokens` and the VLM image path
live in `llm.cpp` / `vlm.cpp`. To change them permanently, edit the
`Args` struct defaults in the corresponding .cpp; to change them for a
single run, pass explicit flags (see "Running a binary manually" below).

## Run

```pwsh
# All tests
ctest --test-dir build -C Release --output-on-failure

# Filter by label
ctest --test-dir build -C Release -L llm            # every LLM test
ctest --test-dir build -C Release -L vlm            # every VLM test
ctest --test-dir build -C Release -L generation     # only generation-mode tests

# Filter by name pattern
ctest --test-dir build -C Release -R qwen3 -V       # verbose output
```

Exit codes from each test binary:

| Code | Meaning |
|---|---|
| `0` | Success (assertion passed) |
| `1` | Init / runtime error (missing model files, QNN init failure, threw exception) |
| `2` | Test assertion failed (e.g. generated fewer than `--min-tokens` tokens) |

## Running a binary manually

For debugging, invoke a test binary outside of `ctest`. Working directory
must be the repo root so that `./modelfiles/<name>/...` resolves. All flags
except `--model` are optional — defaults live in the corresponding .cpp.

```pwsh
# List known LLM model ids
build\bin\Release\llm_test.exe --list-models

# Minimal LLM invocation (uses default prompt / max-tokens / min-tokens)
build\bin\Release\llm_test.exe --model qwen3_4b

# Same with an explicit prompt
build\bin\Release\llm_test.exe --model qwen3_4b --prompt "Hello."

# Minimal VLM invocation (uses default image + prompt)
build\bin\Release\vlm_test.exe --model qwen2_5_vl_7b

# VLM with a custom image
build\bin\Release\vlm_test.exe --model qwen2_5_vl_7b --image path\to\cat.jpg
```

## Design

**Decoupled from examples.** Tests do *not* invoke the interactive example
executables (`qwen3_4b.exe`, `qwen2_5_vl_7b.exe`, etc.). They construct the
pipelines directly via the factories exposed in
[`models/llm_model_registry.h`](../models/llm_model_registry.h) and
[`models/qwen2_5_vl/qwen2_5_vl.h`](../models/qwen2_5_vl/qwen2_5_vl.h). This
keeps the examples pure end-user demos and gives the tests a stable,
non-interactive API.

**Mode dispatch inside each binary.** A single `--mode` flag routes execution
to a specific test function. Today:

```cpp
if (args.mode == "generation") {
    return runGeneration(args, *pipe);
}
```

Adding a new mode is a three-line change: a new `runXxx()` function, a new
branch in the dispatch, and a new block of `add_test()` entries in
`CMakeLists.txt` with `LABELS "llm;xxx"` (or `"vlm;xxx"`).

**Fail fast on missing files.** Before attempting to construct the pipeline,
each test `stat()`s every `.bin` / tokenizer / config path and exits with code
`1` and a clear message if anything is missing. This makes "model files not
staged" easy to distinguish from real runtime failures.

## Adding coverage for a new LLM

1. **Register the model** in [`models/llm_model_registry.h`](../models/llm_model_registry.h)
   (project convention — needed for the interactive examples too).
2. **Add the file layout** to `modelFilesTable()` at the top of
   [`llm.cpp`](llm.cpp). One entry maps the model id to its `modelfiles/`
   subdirectory and `.bin` shard filenames.
3. **Add the model id** to `_LLM_MODELS` in [`CMakeLists.txt`](CMakeLists.txt)
   to register one CTest entry per mode.

Three edits, one per file. No changes to existing tests or examples.

## Adding a new test mode

1. In `llm.cpp` (or `vlm.cpp`) add `static int runXxx(const Args&, Pipeline&)`
   containing the new test logic.
2. Add `if (args.mode == "xxx") return runXxx(args, *pipe);` to the dispatch
   in `main()`.
3. In `CMakeLists.txt` add a second `foreach(_model IN LISTS _LLM_MODELS)`
   loop (or a one-off `add_test()`) using `LABELS "llm;xxx"` so the new tests
   can be filtered with `ctest -L xxx`.

## CI

These tests are **not** run in GitHub CI — the hosted `windows-arm64` runner
has no NPU. CI only compiles the test binaries (when configured with
`-DGENIEX_BUILD_TESTS=ON`, which the build-check workflow does *not* set, so
tests stay off the CI critical path). See
[`.github/workflows/build-check.yml`](../.github/workflows/build-check.yml).
