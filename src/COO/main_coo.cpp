//#include "../cl_defines.hpp"

#include "../utils.hpp"
#include "coo_utils.hpp"
#include "../fast_random.h"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_csr.hpp"
#include "../library_classes/controls.hpp"
#include "coo_matrix_addition.hpp"

#include <vector>
#include <iomanip>
#include <algorithm>
#include <iostream>



void testBitonicSort() {
    // check on different ns

    uint32_t max_size = 32 * 1024 * 1024;
    uint32_t test_num = 1;

    uint32_t n = 0;
    std::vector<uint32_t> rows;
    std::vector<uint32_t> cols;

    FastRandom r(42);

    Controls controls = utils::create_controls();

    for (uint32_t test_iter = 0; test_iter < test_num; ++ test_iter) {
        n = std::abs(r.next()) % max_size;
        std::cout << "for n=" << n << std::endl;
        rows.resize(n);
        cols.resize(n);
        fill_random_matrix(rows, cols);
        uint32_t n_rows = *std::max_element(rows.begin(), rows.end());
        uint32_t n_cols = *std::max_element(cols.begin(), cols.end());

        auto matrix = matrix_coo(controls, n_rows, n_cols, n, rows, cols);
//        sort_arrays(rows, cols);
        coo_utils::check_correctness(matrix.rows_indexes_cpu(), matrix.cols_indexes_cpu());
    }
}


void testAddition() {
    Controls controls = utils::create_controls();

    uint32_t test_size = 42 * 1024;
    std::vector<uint32_t> rowsA(test_size);
    std::vector<uint32_t> colsA(test_size);

    std::vector<uint32_t> rowsB(test_size);
    std::vector<uint32_t> colsB(test_size);

    fill_random_matrix(rowsA, colsA);
    fill_random_matrix(rowsB, colsB);

    uint32_t n_rowsA = *std::max_element(rowsA.begin(), rowsA.end());
    uint32_t n_colsA = *std::max_element(colsA.begin(), colsA.end());

    uint32_t n_rowsB = *std::max_element(rowsB.begin(), rowsB.end());
    uint32_t n_colsB = *std::max_element(colsB.begin(), colsB.end());

    auto matrixA = matrix_coo(controls, n_rowsA, n_colsA, test_size, rowsA, colsA);
    auto matrixB = matrix_coo(controls, n_rowsB, n_colsB, test_size, rowsB, colsB);

    auto matrixC = matrix_coo(controls, std::max(n_rowsA, n_rowsB), std::max(n_colsA, n_colsB), 2 * test_size);
    addition(controls, matrixC, matrixA, matrixB);
}



int main() {
    testAddition();
}


