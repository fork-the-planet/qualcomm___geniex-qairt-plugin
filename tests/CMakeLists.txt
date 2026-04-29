# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# Generation smoke tests for all supported geniex LLM / VLM models.
#
# Two small test binaries drive the pipelines directly (bypassing the
# interactive example executables):
#   * generation_smoke_llm  — dispatch over llm_model_registry
#   * generation_smoke_vlm  — Qwen2.5-VL-7B (other VLMs can be added later)
#
# Per-model `add_test` entries then invoke those binaries with `--model <name>`.
# This keeps CI/CTest decoupled from the examples and makes adding coverage
# for a new LLM a one-line change here once the model is in llm_model_registry.h.
#
# Requirements:
#   * A Snapdragon NPU on the host (cannot run in GitHub CI).
#   * Model files present at <repo>/modelfiles/<name>/ (NOT committed).
#   * For VLM: a small JPEG at <repo>/tests/data/test_image.jpg, or pass
#     -DGENIEX_TEST_IMAGE=<path> at CMake configure time.
#
# Run with:
#     cmake -B build -A ARM64 `
#           -DGENIEX_BUILD_EXAMPLES=ON -DGENIEX_BUILD_VLM=ON `
#           -DGENIEX_BUILD_TESTS=ON
#     cmake --build build --config Release -j
#     ctest --test-dir build -C Release --output-on-failure
#
# Filter by label:
#     ctest --test-dir build -C Release -L llm
#     ctest --test-dir build -C Release -L vlm
# Or by name:
#     ctest --test-dir build -C Release -R qwen3 -V

include(CTest)

# All tests run from the repo root so `./modelfiles/<name>/…` resolves.
set(_GENIEX_TEST_WD "${CMAKE_SOURCE_DIR}")

# ── LLM smoke-test binary ────────────────────────────────────────────────────
add_executable(generation_smoke_llm generation_smoke_llm.cpp)
target_link_libraries(generation_smoke_llm PRIVATE geniex_core geniex-proc)
set_target_properties(generation_smoke_llm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Default generation settings for the LLM tests.
set(_LLM_SMOKE_PROMPT     "Hello, introduce yourself in one short sentence.")
set(_LLM_SMOKE_MAX_TOKENS 32)
set(_LLM_SMOKE_MIN_TOKENS 5)
set(_LLM_SMOKE_TIMEOUT    240)  # seconds; generous for the largest 8B models

# One ctest entry per model id in llm_model_registry.h. Keep this list in
# sync with llm_model_registry() and the modelFilesTable() in
# generation_smoke_llm.cpp.
set(_LLM_SMOKE_MODELS
    qwen3_4b
    qwen3_4b_instruct_2507
    qwen2_5_7b_instruct
    phi_3_5_mini_instruct
    llama_v3_8b_instruct
    llama_v3_elyza_jp_8b
    llama_v3_taide_8b_chat
    llama_v3_1_8b_instruct
    llama_v3_1_sea_lion_3_5_8b_r
    llama_v3_2_1b_instruct
    llama_v3_2_3b_instruct
    falcon_v3_7b_instruct
)

foreach(_model IN LISTS _LLM_SMOKE_MODELS)
    add_test(
        NAME    "llm_${_model}_generate_smoke"
        COMMAND generation_smoke_llm
            --model      "${_model}"
            --prompt     "${_LLM_SMOKE_PROMPT}"
            --max-tokens "${_LLM_SMOKE_MAX_TOKENS}"
            --min-tokens "${_LLM_SMOKE_MIN_TOKENS}"
        WORKING_DIRECTORY "${_GENIEX_TEST_WD}"
    )
    set_tests_properties("llm_${_model}_generate_smoke" PROPERTIES
        TIMEOUT ${_LLM_SMOKE_TIMEOUT}
        LABELS  "smoke;llm"
    )
endforeach()

# ── VLM smoke-test binary ────────────────────────────────────────────────────
# Gated on GENIEX_BUILD_VLM so the test binary only links when the VLM tier
# is actually built.
if(GENIEX_BUILD_VLM)
    # qwen2_5_vl.cpp defines geniex::qwen2_5_vl_7b::makeModel, which is
    # *declared* in qwen2_5_vl.h and called from the inline makePipeline().
    # The per-model .cpp is not compiled into geniex_vlm (it lives with the
    # examples), so the test binary has to pick it up directly — same pattern
    # as models/qwen2_5_vl/CMakeLists.txt.
    add_executable(generation_smoke_vlm
        generation_smoke_vlm.cpp
        ${CMAKE_SOURCE_DIR}/models/qwen2_5_vl/qwen2_5_vl.cpp
    )
    target_link_libraries(generation_smoke_vlm PRIVATE geniex_vlm geniex-proc-vision)
    target_include_directories(generation_smoke_vlm PRIVATE
        ${CMAKE_SOURCE_DIR}/models/qwen2_5_vl
    )
    set_target_properties(generation_smoke_vlm PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # Override with -DGENIEX_TEST_IMAGE=<abs path> at configure time.
    set(GENIEX_TEST_IMAGE "${CMAKE_SOURCE_DIR}/tests/data/test_image.jpg"
        CACHE FILEPATH "Image used by the VLM generation smoke test")

    add_test(
        NAME    "vlm_qwen2_5_vl_7b_generate_smoke"
        COMMAND generation_smoke_vlm
            --model      qwen2_5_vl_7b
            --prompt     "What is in this image?"
            --image      "${GENIEX_TEST_IMAGE}"
            --max-tokens 32
            --min-tokens 5
        WORKING_DIRECTORY "${_GENIEX_TEST_WD}"
    )
    set_tests_properties("vlm_qwen2_5_vl_7b_generate_smoke" PROPERTIES
        TIMEOUT 360  # VLM prefill is heavier; allow more headroom
        LABELS  "smoke;vlm"
    )
endif()
