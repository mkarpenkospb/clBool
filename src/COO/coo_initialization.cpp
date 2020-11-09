#include <vector>
#include <cstddef>
#include <cstdint>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#include <CL/cl.h>
#include "CL/cl2.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

void sort_arrays(std::vector<uint32_t> &rows,
                std::vector<uint32_t> &cols
                ) {

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;
    cl::Program program;

    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        std::cout << "Platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;

        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0]);

        std::cout << "Device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

        // на случай если макрос не скопирует ничего
        // std::ifstream cl_file("../src/cl/convolution.cl");
        std::ifstream cl_file("coo_bitonic_sort.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, cl_string);

        program = cl::Program(context, source);
        unsigned int block_size = 32;
        unsigned int n = rows.size();
        program.build(devices, "-D BLOCK_SIZE=32");

        cl::Buffer a_gpu(context, CL_MEM_READ_ONLY, sizeof(uint32_t) * rows.size());
        cl::Buffer b_gpu(context, CL_MEM_READ_ONLY, sizeof(uint32_t) * cols.size());

        queue.enqueueWriteBuffer(a_gpu, CL_TRUE, 0, sizeof(uint32_t) * rows.size(), rows.data());
        queue.enqueueWriteBuffer(b_gpu, CL_TRUE, 0, sizeof(uint32_t) * cols.size(), cols.data());

        size_t work_group_size = block_size;
        size_t global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;

        std::cout << "\nfinished" << std::endl;
    } catch (cl::Error e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << e.err() << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        }
        throw std::runtime_error(exception.str());
    }
}

