#pragma once

#include <type_traits>
#include <matrix_dcsr.hpp>
#include <cpu_matrices.hpp>
#include "cl_includes.hpp"
#include "controls.hpp"
#include "cstdlib"

namespace utils {

    void compare_matrices(Controls &controls, matrix_dcsr m_gpu, matrix_dcsr_cpu m_cpu);

    void fill_random_buffer(cpu_buffer &buf, uint32_t seed = -1, uint32_t mod = 1);

    void fill_random_buffer(cpu_buffer_f &buf, uint32_t seed = -1);

// https://stackoverflow.com/a/466242
    unsigned int ceil_to_power2(uint32_t v);

// https://stackoverflow.com/a/2681094
    uint32_t round_to_power2(uint32_t x);

    uint32_t calculate_global_size(uint32_t work_group_size, uint32_t n);

    Controls create_controls(const std::string& aosx_name = "");

    std::string error_name(cl_int error);

    void print_gpu_buffer(Controls &controls, const cl::Buffer &buffer, uint32_t size);

    template <typename buf>
    void print_cpu_buffer(const buf &buffer, uint32_t size = -1) {
        uint32_t end = size;
        if (size == -1) end = buffer.size();
        for (uint32_t i = 0; i < end; ++i) {
            std::cout << buffer[i] << ", ";
        }
        std::cout << std::endl;
    }


    template<typename T>
    void compare_buffers(Controls &controls,
                            const cl::Buffer &buffer_g, const std::vector<T, aligned_allocator<T, 64>> &buffer_c, uint32_t size,
                            std::string name = "") {
        using buf = std::vector<T, aligned_allocator<T, 64>>;
        static float epsilon = 0.00001;
        buf cpu_copy(size);
        timer t;
        t.restart();
        cl::Event ev;

        controls.queue.enqueueReadBuffer(buffer_g, CL_TRUE, 0, sizeof(typename buf::value_type) * cpu_copy.size(),
                                         cpu_copy.data(), nullptr, &ev);
        ev.wait();
        double time = t.elapsed();
        if (DEBUG_ENABLE) *logger << "read buffer in " << time << " \n";

        for (uint32_t i = 0; i < size; ++i) {
            typename buf::value_type d = cpu_copy[i] - buffer_c[i];
            if (d >  epsilon || d < -epsilon) {
                uint32_t start = std::max(0, (int) i - 10);
                uint32_t stop = std::min(size, i + 10);
                for (uint32_t j = start; j < stop; ++j) {
                    std::cout << j << ": (" << cpu_copy[j] << ", " << buffer_c[j] << "), ";
                }
                std::cout << std::endl;
                throw std::runtime_error("buffers for " + name + " are different");
            }
        }
        if (DEBUG_ENABLE) *logger << "buffers are equal\n";
    }

    void program_handler(const cl::Error &e, const cl::Program &program,
                         const cl::Device &device, const std::string &name);

    void show_devices();
//    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, const coo_utils::matrix_dcsr_cpu &m, uint32_t size);
}