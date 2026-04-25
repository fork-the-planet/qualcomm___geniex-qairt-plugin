# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: Apache-2.0

foreach(EXAMPLE_NAME llama_v3_8b_instruct llama_v3_elyza_jp_8b llama_v3_taide_8b_chat)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}_example.cpp)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE geniex_core geniex-proc)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()
