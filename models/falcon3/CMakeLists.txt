add_executable(falcon3_7b_aihub falcon3_7b_aihub_example.cpp)
target_link_libraries(falcon3_7b_aihub PRIVATE geniex_core geniex-proc)
set_target_properties(falcon3_7b_aihub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
