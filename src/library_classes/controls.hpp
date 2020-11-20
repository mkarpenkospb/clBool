#pragma once

#include "../cl_defines.hpp"
#include <string>
#include <iostream>
#include <sstream>

// TODO: in opencl 2.0 we have DeviceCommandQueue class so what about opencl 2.0?

struct Controls {
    const cl::Device device;
    const cl::Context context;
    cl::CommandQueue queue;
    const uint32_t block_size = uint32_t(256);

    Controls(cl::Device device) :
    device(device)
    , context(cl::Context(device))
    , queue(cl::CommandQueue(context))
    {}
};

inline Controls create_controls() {
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
