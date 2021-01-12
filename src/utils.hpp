#pragma once

#include <cstdint>
#include "cl_includes.hpp"
#include "library_classes/controls.hpp"

namespace utils {
    using cpu_buffer = std::vector<uint32_t>;

// https://stackoverflow.com/a/466242
    unsigned int ceil_to_power2(uint32_t v);

// https://stackoverflow.com/a/2681094
    uint32_t round_to_power2(uint32_t x);

    uint32_t calculate_global_size(uint32_t work_group_size, uint32_t n);

    Controls create_controls();

    std::string error_name(cl_int error);

    void print_gpu_buffer(Controls &controls, const cl::Buffer &buffer, uint32_t size);

    void print_cpu_buffer(const cpu_buffer& buffer);

    void compare_buffers(Controls &controls, const cl::Buffer &buffer_g, const cpu_buffer& buffer_c, uint32_t size);
}