//
// Created by mkarp on 08.11.2020.
//
#include "convolution.hpp"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#include <CL/cl.h>
#include "CL/cl2.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

void convoluion(const std::vector<value_type> &a,
                const std::vector<value_type> &b,
                std::vector<value_type> &c,
                int n,
                int m) {

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
        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, cl_string);

        program = cl::Program(context, source);

        std::stringstream options;
        size_t const block_size = 32;
        options << "-D BLOCK_SIZE=32 -D CONV_SIZE=" << m;

        program.build(devices, options.str().c_str());

        cl::Buffer a_gpu(context, CL_MEM_READ_ONLY, sizeof(value_type) * a.size());
        cl::Buffer b_gpu(context, CL_MEM_READ_ONLY, sizeof(value_type) * b.size());
        cl::Buffer c_gpu(context, CL_MEM_WRITE_ONLY, sizeof(value_type) * c.size());

        queue.enqueueWriteBuffer(a_gpu, CL_TRUE, 0, sizeof(value_type) * a.size(), a.data());
        queue.enqueueWriteBuffer(b_gpu, CL_TRUE, 0, sizeof(value_type) * b.size(), b.data());

        size_t work_group_size = block_size;
        size_t global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;

        cl::Kernel convolution_kernel(program, "convolution");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int> convolution(convolution_kernel);
        cl::EnqueueArgs eargs(queue, cl::NDRange(global_work_size, global_work_size),
                              cl::NDRange(work_group_size, work_group_size));

        cl::Event event = convolution(eargs, a_gpu, b_gpu, c_gpu, (int) n);
        event.wait();
        queue.enqueueReadBuffer(c_gpu, CL_TRUE, 0, sizeof(value_type) * c.size(), c.data());

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

void print_matrix(const std::vector<value_type> &a, int n) {
    std::cout << std::endl;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::cout << a[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

auto make_fit(int n) {
    return [n](int idx) {return idx >= 0 && idx < n;};
}

void check_correctness(const std::vector<value_type> &a,
                       const std::vector<value_type> &b,
                       const std::vector<value_type> &c,
                       int n,
                       int m) {

    auto fit = make_fit(n);

    int m_half = m / 2;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            value_type acc = 0;
            for (int im = -m_half; im <= m_half; ++im) {
                for (int jm = -m_half; jm <= m_half; ++jm) {
                    int i_read_a = i + im;
                    int j_read_a = j + jm;
                    int i_read_b = im + m_half;
                    int j_read_b = jm + m_half;
                    acc += fit(i_read_a) && fit(j_read_a) ?
                            a[i_read_a * n + j_read_a] * b[i_read_b * m + j_read_b] : 0;
                }
            }
            if (std::abs(acc - c[i * n + j]) > 0.00001) {
                std::stringstream exception;
                exception << "incorrect result with expected "<< std::fixed << std::setprecision(5) <<  acc << " , got " <<  c[i * n + j] << "\n";
                throw std::runtime_error(exception.str());
            }
        }
    }
}

void parse_input(const std::string& filename,
                 std::vector<value_type>& a,
                 std::vector<value_type>& b,
                 int& n, int& m) {
    std::ifstream file(filename);

    if (!file) {
        std::stringstream exception;
        exception << "cannot open " << filename << " for read\n";
        throw std::runtime_error(exception.str());
    }

    file >> n;
    file >> m;

    if (!(n >= 1 && n <= 2014)) {
        std::stringstream exception;
        exception << "first matrix size does not satisfy 1 << n << 1024 with n=" << n << "\n";
        throw std::invalid_argument(exception.str());
    }

    if (!(m >= 1 && m <= 9)) {
        std::stringstream exception;
        exception << "second matrix size does not satisfy 1 << m << 9 with m=" << m << "\n";
        throw std::invalid_argument(exception.str());
    }

    if (m % 2 != 1) {
        std::stringstream exception;
        exception << "second matrix size is not odd with m=" << m << "\n";
        throw std::invalid_argument(exception.str());
    }

    value_type elem;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            file >> elem;
            a.push_back(elem);
        }
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            file >> elem;
            b.push_back(elem);
        }
    }

    file.close();
}

void write_result(const std::string& filename,
                 const std::vector<value_type>& c,
                 int n) {
    std::ofstream file(filename);
    if (!file) {
        std::stringstream exception;
        exception << "cannot open " << filename << " for write\n";
        throw std::runtime_error(exception.str());
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            file << c[i * n + j] << " ";
        }
        file << "\n";
    }
    file.close();
}

void fill_random_matrix(std::vector<value_type>& a, int nxn) {
    for (size_t i = 0; i < nxn; ++i) {
        a[i] = (rand()  % 10) * 1.0/ 10;
    }
}