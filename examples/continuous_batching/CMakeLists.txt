# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# ── examples/continuous_batching ─────────────────────────────────────────────
#
# Demos of continuous batching on top of geniex_core. The shared CB library
# is header-only (under common/cb/), so each model subfolder just adds its
# own executable that includes `cb/cb.h`.
#
# To add a new model demo, create a new subdirectory with its own
# CMakeLists.txt and add it below. See README.md for the full recipe.

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/qwen3)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/llama3_2)
