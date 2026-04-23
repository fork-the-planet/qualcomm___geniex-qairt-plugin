add_executable(falcon3_7b falcon3_7b_example.cpp)
target_link_libraries(falcon3_7b PRIVATE geniex_core geniex-proc)
set_target_properties(falcon3_7b PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
