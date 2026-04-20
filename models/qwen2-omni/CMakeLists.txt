add_executable(qwen2_omni
    qwen2_omni_example.cpp
    qwen2_omni.cpp
)
target_include_directories(qwen2_omni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(qwen2_omni PRIVATE geniex_vlm)
set_target_properties(qwen2_omni PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
