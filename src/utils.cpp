#include "utils.hpp"

#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>


// https://stackoverflow.com/a/466242
unsigned int ceil_to_power2(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

// https://stackoverflow.com/a/2681094
uint32_t round_to_power2 (uint32_t x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

uint32_t calculate_global_size(uint32_t work_group_size, uint32_t n) {
    return (n + work_group_size - 1) / work_group_size * work_group_size;
}

Controls create_controls() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;
    cl::Program program;
    cl::Device device;
    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        return Controls(devices[0]);

    } catch (const cl::Error& e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << e.err() << "\n";
        throw std::runtime_error(exception.str());
    }
}