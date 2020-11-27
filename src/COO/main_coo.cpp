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



void testMatrixInitialisation() {
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
        coo_utils::fill_random_matrix(rows, cols);
        uint32_t n_rows = *std::max_element(rows.begin(), rows.end());
        uint32_t n_cols = *std::max_element(cols.begin(), cols.end());

        auto matrix = matrix_coo(controls, n_rows, n_cols, n, rows, cols);
//        sort_arrays(rows, cols);
        coo_utils::check_correctness(matrix.rows_indexes_cpu(), matrix.cols_indexes_cpu());
    }
}
using coo_utils::matrix_cpp_cpu;

void testBitonicSort() {
    Controls controls = utils::create_controls();

    uint32_t size = 42 * 1024;
    std::vector<uint32_t> rows_cpu(size);
    std::vector<uint32_t> cols_cpu(size);

    std::vector<uint32_t> rows_from_gpu(size);
    std::vector<uint32_t> cols_from_gpu(size);

    coo_utils::fill_random_matrix(rows_cpu, cols_cpu);

    matrix_cpp_cpu m_cpu;

    coo_utils::form_cpu_matrix(m_cpu, rows_cpu, cols_cpu);

    std::sort(m_cpu.begin(), m_cpu.end());

    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, m_cpu);

    cl::Buffer rows_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
    cl::Buffer cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

    controls.queue.enqueueWriteBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_cpu.data());
    controls.queue.enqueueWriteBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_cpu.data());

    sort_arrays(controls, rows_gpu, cols_gpu, size);

    controls.queue.enqueueReadBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_from_gpu.data());
    controls.queue.enqueueReadBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_from_gpu.data());

    if (rows_from_gpu == rows_cpu || cols_from_gpu == cols_cpu) {
        std::cout << "correct" << std::endl;
    } else {
        std::cerr << "incorrect" << std::endl;
    }
}


//void testAddition() {
//    Controls controls = utils::create_controls();
//
//    uint32_t test_size = 42 * 1024;
//    std::vector<uint32_t> rowsA(test_size);
//    std::vector<uint32_t> colsA(test_size);
//
//    std::vector<uint32_t> rowsB(test_size);
//    std::vector<uint32_t> colsB(test_size);
//
//    fill_random_matrix(rowsA, colsA);
//    fill_random_matrix(rowsB, colsB);
//
//    uint32_t n_rowsA = *std::max_element(rowsA.begin(), rowsA.end());
//    uint32_t n_colsA = *std::max_element(colsA.begin(), colsA.end());
//
//    uint32_t n_rowsB = *std::max_element(rowsB.begin(), rowsB.end());
//    uint32_t n_colsB = *std::max_element(colsB.begin(), colsB.end());
//
//    auto matrixA = matrix_coo(controls, n_rowsA, n_colsA, test_size, rowsA, colsA);
//    auto matrixB = matrix_coo(controls, n_rowsB, n_colsB, test_size, rowsB, colsB);
//
//    auto matrixC = matrix_coo(controls, std::max(n_rowsA, n_rowsB), std::max(n_colsA, n_colsB), 2 * test_size);
//    addition(controls, matrixC, matrixA, matrixB);
//}



int main() {
    testBitonicSort();
}


