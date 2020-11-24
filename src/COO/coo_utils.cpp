#include <cstdint>
#include "coo_utils.hpp"
#include "../library_classes/matrix_coo.hpp"
#include <vector>
#include <iostream>

namespace coo_utils {

    void check_correctness(const std::vector<uint32_t> &rows, const std::vector<uint32_t> &cols) {
        uint32_t n = rows.size();
        for (uint32_t i = 1; i < n; ++i) {
            if (rows[i] < rows[i - 1] || (rows[i] == rows[i - 1] && cols[i] < cols[i - 1])) {
                uint32_t start = i < 10 ? 0 : i - 10;
                uint32_t stop = i >= n - 10 ? n : i + 10;
                for (uint32_t k = start; k < stop; ++k) {
                    //TODO: all type of streams as parameter!!!!!!!!!
                    std::cout << k << ": (" << rows[k] << ", " << cols[k] << "), ";
                }
                std::cout << std::endl;
                throw std::runtime_error("incorrect result!");
            }
        }
        std::cout << "check finished, probably correct\n";
    }

//    matrix_coo generate_random_matrix() {
//
//    }
//
}