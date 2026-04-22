add_executable(qwen2_5_vl_7b qwen2_5_vl_7b_example.cpp)
target_link_libraries(qwen2_5_vl_7b PRIVATE geniex_core geniex-proc)
set_target_properties(qwen2_5_vl_7b PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
