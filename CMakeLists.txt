cmake_minimum_required (VERSION 3.1)
project(sparse-boolean-matrix-operations)

# у меня не работает юбез явного указания путей
set(OpenCL_LIBRARY "C:/Windows/System32/OpenCL.dll")
set(OpenCL_INCLUDE_DIR_LIBRARY "C:/Windows/System32/")
set(OpenCL_INCLUDE_DIR "libs/clew")

set(Library_classes "src/library_classes")


# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
#add_executable (sparse-boolean-matrix-operations src/example/cl/clion_defines.cl src/example/convolution.cpp)
#target_link_libraries (sparse-boolean-matrix-operations ${OpenCL_LIBRARY})


add_executable(sparse-boolean-matrix-operations
        src/library_classes/matrix_coo.cpp
        src/coo/main_coo.cpp
        src/coo/coo_matrix_addition.cpp
        src/coo/cl/clion_defines.cl
        src/coo/coo_initialization.cpp
        src/utils.cpp
        src/coo/cl/coo_bitonic_sort.cl
        src/coo/cl/coo_kronecker.cl
        src/coo/cl/merge_path.cl
        src/coo/coo_utils.cpp
        src/coo/cl/prepare_positions.cl
        src/coo/cl/prefix_sum.cl
        src/coo/cl/set_positions.cl
        src/coo/cl/count_workload.cl
        src/coo/cl/heap_merge.cl
        src/coo/cl/bitonic_esc.cl
        src/coo/cl/merge_rows.cl
        src/coo/coo_kronecker_product.cpp
        src/coo/coo_matrix_multiplication.cpp

        src/coo/test/coo_addition_test.cpp
        src/coo/test/coo_bitonic_test.cpp
        src/coo/test/coo_reduce_duplicates_test.cpp
        src/coo/test/coo_kronecker_test.cpp
        src/coo/test/coo_to_dcsr_test.cpp

        )

target_link_libraries (sparse-boolean-matrix-operations ${OpenCL_LIBRARY})
# у меня этот макрос через раз отрабатывает, вроде файл поменяла, а он не скопировался
# copy cl files to binary folder
FILE(GLOB CL_FILES src/coo/cl/*.cl)
foreach(cl_file ${CL_FILES})
    add_custom_command(
    TARGET sparse-boolean-matrix-operations
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy ${cl_file} ${PROJECT_BINARY_DIR}/.
)
endforeach(cl_file)

