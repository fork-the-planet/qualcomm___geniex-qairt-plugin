add_executable(qwen3_cb qwen3_cb_example.cpp)
target_link_libraries(qwen3_cb PRIVATE geniex_core geniex-proc)
set_target_properties(qwen3_cb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
