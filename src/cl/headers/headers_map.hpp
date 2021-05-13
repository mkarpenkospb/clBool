#pragma once
#include <unordered_map>
#include "prefix_sum.h"


struct KernelSource {
    const char* kernel;
    size_t length;
};

static const std::unordered_map<std::string, KernelSource> HeadersMap = {
        {"prefix_sum", {prefix_sum_kernel, prefix_sum_kernel_length}},
};