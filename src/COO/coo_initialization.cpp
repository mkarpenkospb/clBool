#include <vector>
#include <cstddef>
#include <cstdint>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#include <CL/cl.h>
#include "CL/cl2.hpp"
#include "../utils.hpp"
#include "../fast_random.h"
#include "coo_initialization.hpp"

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
    cl::Device device;

    try {

        // ----------------------------------- opencl settings -----------------------------------------------------



        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        std::cout << "Platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;

        cl::Context context(devices);
        device = devices[0];
        cl::CommandQueue queue(context, device);

        std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // на случай если макрос не скопирует ничего
        std::ifstream cl_file("../src/COO/cl/coo_bitonic_sort.cl");
//        std::ifstream cl_file("coo_bitonic_sort.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, cl_string);

        program = cl::Program(context, source);
        uint32_t block_size = 256;
        uint32_t n = rows.size();
        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(devices, options.str().c_str());



        // -------------------------------------- set kernels and its arguments -----------------------------------------------------


        cl::Buffer rows_gpu(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n);
        cl::Buffer cols_gpu(context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n);

        queue.enqueueWriteBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * n, rows.data());
        queue.enqueueWriteBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * n, cols.data());

        uint32_t work_group_size = block_size;
        // a bitonic sort needs 2 time less threads than values in array to sort
        uint32_t global_work_size = calculate_global_size(work_group_size, round_to_power2(n));

        cl::Kernel coo_bitonic_begin_kernel(program, "local_bitonic_begin");
        cl::Kernel coo_bitonic_global_step_kernel(program, "bitonic_global_step");
        cl::Kernel coo_bitonic_end_kernel(program, "bitonic_local_endings");

        cl::KernelFunctor<cl::Buffer, cl::Buffer, uint32_t>
                coo_bitonic_begin(coo_bitonic_begin_kernel);
        cl::KernelFunctor<cl::Buffer, cl::Buffer, uint32_t, uint32_t, uint32_t>
                coo_bitonic_global_step(coo_bitonic_global_step_kernel);
        cl::KernelFunctor<cl::Buffer, cl::Buffer, uint32_t>
                coo_bitonic_end(coo_bitonic_end_kernel);

        cl::EnqueueArgs eargs(queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));
        // ----------------------------------------------- main cycle -----------------------------------------------
        coo_bitonic_begin(eargs, rows_gpu, cols_gpu, n);

        uint32_t segment_length = work_group_size * 2 * 2;
//        // TODO : power function for unsigned?
        uint32_t outer = ceil_to_power2(ceil(n * 1.0 / (work_group_size * 2)));

        while (outer != 1) {
            coo_bitonic_global_step(eargs, rows_gpu, cols_gpu, segment_length, 1, n);
            for (unsigned int i = segment_length / 2; i > work_group_size * 2;  i >>= 1) {
                coo_bitonic_global_step(eargs, rows_gpu, cols_gpu, i, 0, n);
            }
            coo_bitonic_end(eargs, rows_gpu, cols_gpu, n);
            outer >>= 1;
            segment_length <<= 1;
        }
//        cl::Event waitRead1;
//        cl::Event waitRead2;
        queue.enqueueReadBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * n, rows.data());
        queue.enqueueReadBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * n, cols.data());
//        waitRead1.wait();
//        waitRead2.wait();
//        check_correctness(rows, cols);
        std::cout << "\nfinished" << std::endl;
    } catch (cl::Error e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << e.err() << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        }
        throw std::runtime_error(exception.str());
    }
}

void check_correctness(const std::vector<uint32_t>& rows, const std::vector<uint32_t>& cols) {
    uint32_t n = rows.size();
    for (uint32_t i = 1; i < n; ++i) {
        if (rows[i] < rows[i - 1] || (rows[i] == rows[i - 1] && cols[i] < cols[i-1])) {
            uint32_t start = i < 10 ? 0 : i - 10;
            uint32_t stop = i >=  n  - 10 ? n : i + 10;
            for (uint32_t k = start; k < stop; ++k) {
                std::cout << k  << ": (" << rows[k] << ", " << cols[k] << "), ";
            }
            std::cout << std::endl;
            throw std::runtime_error("incorrect result!");
        }
    }
    std::cout << "check finished, probably correct\n";
}

void fill_random_matrix(std::vector<uint32_t>& rows, std::vector<uint32_t>& cols) {
    uint32_t n = rows.size();
    FastRandom r(n);
    for (uint32_t i = 0 ; i < n; ++i) {
        rows[i] = r.next() % 1024;
        cols[i] = r.next() % 1024;
    }
}