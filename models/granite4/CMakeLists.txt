add_executable(granite4_micro granite4_micro_example.cpp)
target_link_libraries(granite4_micro PRIVATE geniex_core geniex-proc)
set_target_properties(granite4_micro PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_executable(granite4_micro_pipeline granite4_micro_pipeline.cpp)
target_link_libraries(granite4_micro_pipeline PRIVATE geniex_core geniex-proc)
set_target_properties(granite4_micro_pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
