#include <cstddef>
#include <iostream>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#include "coo_initialization.hpp"
#include "../fast_random.h"


int main() {
    // check on different ns

    uint32_t max_size = 32 * 1024 * 1024;
    uint32_t test_num = 6;

    uint32_t n = 0;
    std::vector<uint32_t> rows;
    std::vector<uint32_t> cols;

    FastRandom r(42);

    for (uint32_t test_iter = 0; test_iter < test_num; ++ test_iter) {
        n = std::abs(r.next()) % max_size;
        std::cout << "for n=" << n << std::endl;
        rows.resize(n);
        cols.resize(n);
        fill_random_matrix(rows, cols);
        sort_arrays(rows, cols);
        check_correctness(rows, cols);
    }

}


