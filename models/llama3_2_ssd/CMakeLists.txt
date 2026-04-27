# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

add_executable(llama3_2_3b_ssd
    llama3_2_ssd_example.cpp
    ssd_model.cpp
)
target_include_directories(llama3_2_3b_ssd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama3_2_3b_ssd PRIVATE geniex_core geniex-proc)
set_target_properties(llama3_2_3b_ssd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
