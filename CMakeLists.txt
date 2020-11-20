cmake_minimum_required (VERSION 3.1)
project(sparse-boolean-matrix-operations)

# у меня не работает юбез явного указания путей
set(OpenCL_LIBRARY "C:/Windows/System32/OpenCL.dll")
set(OpenCL_INCLUDE_DIR "libs/clew")

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
#add_executable (sparse-boolean-matrix-operations src/example/cl/clion_defines.cl src/example/convolution.cpp)
#target_link_libraries (sparse-boolean-matrix-operations ${OpenCL_LIBRARY})


add_executable(sparse-boolean-matrix-operations
        src/COO/main_coo.cpp
        src/COO/cl/clion_defines.cl
        src/COO/coo_initialization.cpp
        src/utils.cpp
        src/COO/cl/coo_bitonic_sort.cl
        src/COO/cl/coo_kronecker.cl )
target_link_libraries (sparse-boolean-matrix-operations ${OpenCL_LIBRARY})


# у меня этот макрос через раз отрабатывает, вроде файл поменяла, а он не скопировался
# copy cl files to binary folder
#FILE(GLOB CL_FILES src/example/cl/*.cl)
#foreach(cl_file ${CL_FILES})
#    add_custom_command(
#    TARGET sparse-boolean-matrix-operations
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND}
#    ARGS -E copy ${cl_file} ${PROJECT_BINARY_DIR}/.
#)
#endforeach(cl_file)

