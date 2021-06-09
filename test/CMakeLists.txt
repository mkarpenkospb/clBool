project(test)

add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
message(${gtest_SOURCE_DIR})

add_executable(test_operations src/operations.cpp test_operations.cpp src/clBool_tests.cpp)
target_link_libraries(test_operations gtest gtest_main clbool)

add_executable(test_procedures src/procedures.cpp test_procedures.cpp src/clBool_tests.cpp)
target_link_libraries(test_procedures gtest gtest_main clbool)

add_executable(test_errors test_errors.cpp src/clBool_tests.cpp)
target_link_libraries(test_errors gtest gtest_main clbool)

add_executable(test_conversions test_conversions.cpp src/conversions.cpp src/clBool_tests.cpp)
target_link_libraries(test_conversions gtest gtest_main clbool)


include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(
        "../libs/clew"
        "../src/core"
        "../src/common"
        "../src/cl/headers"
        "../libs/utils"
)

