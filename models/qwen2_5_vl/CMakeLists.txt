add_executable(qwen2_5_vl_7b
    qwen2_5_vl_7b_example.cpp
    qwen2_5_vl.cpp
)
target_include_directories(qwen2_5_vl_7b PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(qwen2_5_vl_7b PRIVATE geniex_vlm geniex-proc-vision)
set_target_properties(qwen2_5_vl_7b PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
