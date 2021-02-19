#pragma once

#include <type_traits>
#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_dcsr.hpp"
#include "../library_classes/cpu_matrices.hpp"


namespace utils {
    template<typename, typename = std::void_t<>>
    struct proper_container : std::false_type { };

    template<typename T>
    struct proper_container<T, std::void_t<typename T::iterator,
            typename T::const_iterator,
            typename T::value_type>> : std::true_type { };

    void compare_matrices(Controls &controls, matrix_dcsr m_gpu, matrix_dcsr_cpu m_cpu);

    using cpu_buffer = std::vector<uint32_t>;

    void fill_random_buffer(cpu_buffer &buf, uint32_t seed = -1);

    void fill_random_buffer(cpu_buffer_f &buf, uint32_t seed = -1);

// https://stackoverflow.com/a/466242
    unsigned int ceil_to_power2(uint32_t v);

// https://stackoverflow.com/a/2681094
    uint32_t round_to_power2(uint32_t x);

    uint32_t calculate_global_size(uint32_t work_group_size, uint32_t n);

    Controls create_controls();

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

    template<typename buf>
    std::enable_if_t<proper_container<buf>::value>
            compare_buffers(Controls &controls,
                            const cl::Buffer &buffer_g, const buf &buffer_c, uint32_t size,
                            std::string name = "") {
                static float epsilon = 0.00001;
        buf cpu_copy(size);
        controls.queue.enqueueReadBuffer(buffer_g, CL_TRUE, 0, sizeof(typename buf::value_type) * cpu_copy.size(),
                                         cpu_copy.data());
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
        std::cout << "buffers are equal" << std::endl;
    }

    void program_handler(const cl::Error &e, const cl::Program &program,
                         const cl::Device &device, const std::string &name);

    void show_devices();
//    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, const coo_utils::matrix_dcsr_cpu &m, uint32_t size);
}