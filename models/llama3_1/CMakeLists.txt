foreach(EXAMPLE_NAME llama3_1_8b sea_lion_8b)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_NAME}_example.cpp)
    target_link_libraries(${EXAMPLE_NAME} PRIVATE geniex_core geniex-proc)
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()
