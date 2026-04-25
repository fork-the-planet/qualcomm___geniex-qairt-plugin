# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: Apache-2.0

add_executable(llama3_2_cb llama3_2_cb_example.cpp)

# Include paths:
#   - examples/continuous_batching/common    → cb/*.h shared headers
#   - examples/continuous_batching/llama3_2  → llama3_2_cb.h (current dir)
# Specs (llama3_2_1b / llama3_2_3b) come from models/llama3_2/llama3_2.h,
# whose include dir is already added to geniex_core.
target_include_directories(llama3_2_cb PRIVATE
    ${CMAKE_SOURCE_DIR}/examples/continuous_batching/common
    ${CMAKE_SOURCE_DIR}/examples/continuous_batching/llama3_2
)

target_link_libraries(llama3_2_cb PRIVATE geniex_core geniex-proc)

set_target_properties(llama3_2_cb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
