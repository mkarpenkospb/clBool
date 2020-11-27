#include <cstdint>
#include "coo_utils.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../fast_random.h"
#include <vector>
#include <iostream>
#include <algorithm>

namespace coo_utils {

    void fill_random_matrix(std::vector<uint32_t>& rows, std::vector<uint32_t>& cols) {
        uint32_t n = rows.size();
        FastRandom r(n);
        for (uint32_t i = 0 ; i < n; ++i) {
            // чтобы нулей не было
            rows[i] = r.next() % 1024 + 1;
            cols[i] = r.next() % 1024 + 1;
        }
    }

    void form_cpu_matrix(matrix_cpp_cpu& matrix_out, std::vector<uint32_t>& rows, std::vector<uint32_t>& cols) {
        matrix_out.resize(rows.size());
        std::transform(rows.begin(), rows.end(), cols.begin(), matrix_out.begin(),
                       [](uint32_t i, uint32_t j) -> coordinates {return {i, j};});

    }

    void get_vectors_from_cpu_matrix(std::vector<uint32_t>& rows_out, std::vector<uint32_t>& cols_out, matrix_cpp_cpu& matrix) {
        rows_out.resize(matrix.size());
        cols_out.resize(matrix.size());

        std::for_each(matrix.begin(), matrix.end(),
                      [&rows_out, &cols_out](coordinates& crd) {
            static uint32_t i = 0;
            rows_out[i] = crd.first;
            cols_out[i] = crd.second;
            ++i;} );
    }


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