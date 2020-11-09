cmake_minimum_required (VERSION 3.1)
project(01-opencl-mkarpenkospb)

# у меня не работает юбез явного указания путей
#set(OpenCL_LIBRARY "C:/Windows/System32/OpenCL.dll")
#set(OpenCL_INCLUDE_DIR "libs/clew")

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
add_executable (01-opencl-mkarpenkospb src/main.cpp src/cl/convolution.cl src/cl/clion_defines.cl src/convolution.cpp)
target_link_libraries (01-opencl-mkarpenkospb ${OpenCL_LIBRARY})


# у меня этот макрос через раз отрабатывает, вроде файл поменяла, а он не скопировался
# copy cl files to binary folder
FILE(GLOB CL_FILES src/cl/*.cl)
foreach(cl_file ${CL_FILES})
    add_custom_command(
    TARGET 01-opencl-mkarpenkospb
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy ${cl_file} ${PROJECT_BINARY_DIR}/.
)
endforeach(cl_file)

