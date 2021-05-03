project(test)

add_subdirectory(lib)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
message(${gtest_SOURCE_DIR})

add_executable(test
        # test sources

        src/operations.cpp
        src/procedures.cpp

        test_procedures.cpp
        test_operations.cpp)
target_link_libraries(test gtest gtest_main clBool)

include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

include_directories(
        "../libs/clew"
        "../src/core"
        "../src/common"
        "../src/cl/headers"
        "../libs/utils"
)


