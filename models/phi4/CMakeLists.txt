add_executable(phi4 phi4_example.cpp)
target_link_libraries(phi4 PRIVATE geniex_core geniex-proc)
set_target_properties(phi4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
