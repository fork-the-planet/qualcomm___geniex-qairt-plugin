# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_executable(auto_llm auto_llm.cpp)
target_link_libraries(auto_llm PRIVATE geniex_core geniex-proc)
set_target_properties(auto_llm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
