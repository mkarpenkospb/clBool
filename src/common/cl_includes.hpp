#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#define FPGA

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__))
#define WIN
#endif

#include "CL/cl.h"
#include "CL/opencl.hpp"