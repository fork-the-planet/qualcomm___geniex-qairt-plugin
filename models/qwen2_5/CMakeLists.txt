add_executable(qwen2_5_7b_instruct qwen2_5_7b_instruct_example.cpp)
target_link_libraries(qwen2_5_7b_instruct PRIVATE geniex_core geniex-proc)
set_target_properties(qwen2_5_7b_instruct PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
