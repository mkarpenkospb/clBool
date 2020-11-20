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


