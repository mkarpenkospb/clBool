//#include "../cl_defines.hpp"

#include "../utils.hpp"
#include "coo_utils.hpp"
#include "../fast_random.h"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_csr.hpp"
#include "../library_classes/controls.hpp"
#include "coo_matrix_addition.hpp"
#include "coo_kronecker_product.hpp"

#include <vector>
#include <iomanip>
#include <algorithm>
#include <iostream>

using coo_utils::matrix_cpp_cpu;

void testBitonicSort() {
    Controls controls = utils::create_controls();

    uint32_t size = 15;
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

    if (rows_from_gpu == rows_cpu && cols_from_gpu == cols_cpu) {
        std::cout << "correct sort" << std::endl;
    } else {
        std::cerr << "incorrect sort" << std::endl;
    }
}


void testReduceDuplicates() {

    Controls controls = utils::create_controls();


    uint32_t size = 10374663;

    // -------------------- create indices ----------------------------

    std::vector<uint32_t> rows_cpu(size);
    std::vector<uint32_t> cols_cpu(size);

    std::vector<uint32_t> rows_from_gpu(size);
    std::vector<uint32_t> cols_from_gpu(size);

    coo_utils::fill_random_matrix(rows_cpu, cols_cpu, 1043);

    // -------------------- create and sort cpu matrix ----------------------------
    matrix_cpp_cpu m_cpu;
    coo_utils::form_cpu_matrix(m_cpu, rows_cpu, cols_cpu);
    std::sort(m_cpu.begin(), m_cpu.end());

    // -------------------- create and sort gpu buffers ----------------------------

    cl::Buffer rows_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
    cl::Buffer cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

    controls.queue.enqueueWriteBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_cpu.data());
    controls.queue.enqueueWriteBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_cpu.data());

    sort_arrays(controls, rows_gpu, cols_gpu, size);

    // ------------------ now reduce cpu matrix and read result in vectors ------------------------

    std::cout << "\nmatrix cpu before size: " << m_cpu.size() << std::endl;
    m_cpu.erase(std::unique(m_cpu.begin(), m_cpu.end()), m_cpu.end());
    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, m_cpu);
    std::cout << "\nmatrix cpu after size: " << m_cpu.size() << std::endl;

    // ------------------ now reduce gpu buffers and read in vectors ------------------------
    size_t new_size;
    reduce_duplicates(controls, rows_gpu, cols_gpu, reinterpret_cast<uint32_t &>(new_size), size);

    rows_from_gpu.resize(new_size);
    cols_from_gpu.resize(new_size);

    controls.queue.enqueueReadBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * new_size, rows_from_gpu.data());
    controls.queue.enqueueReadBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * new_size, cols_from_gpu.data());

    if (rows_from_gpu == rows_cpu && cols_from_gpu == cols_cpu) {
        std::cout << "correct reduce" << std::endl;
    } else {
        std::cerr << "incorrect reduce" << std::endl;
    }
}

void testMatrixAddition() {
    Controls controls = utils::create_controls();

    matrix_cpp_cpu matrix_res_cpu;
    // first argument is pseudo size (size before reducing duplicates after random)
    // second is the maximum possible matrix size
    matrix_cpp_cpu matrix_a_cpu = coo_utils::generate_random_matrix_cpu(45726, 1056);
    matrix_cpp_cpu matrix_b_cpu = coo_utils::generate_random_matrix_cpu(667312, 3526);

    matrix_coo matrix_res_gpu;
    matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu);
    matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu);

    coo_utils::matrix_addition_cpu(matrix_res_cpu, matrix_a_cpu, matrix_b_cpu);

    matrix_addition(controls, matrix_res_gpu, matrix_a_gpu, matrix_b_gpu);

    std::vector<uint32_t> rows_cpu;
    std::vector<uint32_t> cols_cpu;

    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, matrix_res_cpu);

    if (matrix_res_gpu.rows_indices_cpu() == rows_cpu && matrix_res_gpu.cols_indices_cpu() == cols_cpu) {
        std::cout << "correct addition" << std::endl;
    } else {
        std::cerr << "incorrect addition" << std::endl;
    }

}

void testKronecker() {
    Controls controls = utils::create_controls();

    matrix_cpp_cpu matrix_res_cpu;
    // first argument is nnz (nnz before reducing duplicates after random)
    // second is the maximum possible matrix size
    matrix_cpp_cpu matrix_a_cpu = coo_utils::generate_random_matrix_cpu(2452, 379);
    matrix_cpp_cpu matrix_b_cpu = coo_utils::generate_random_matrix_cpu(7553, 395);

    matrix_coo matrix_res_gpu;
    matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu);
    matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu);

    coo_utils::kronecker_product_cpu(matrix_res_cpu, matrix_a_cpu, matrix_b_cpu);

    kronecker_product(controls, matrix_res_gpu, matrix_a_gpu, matrix_b_gpu);

    std::vector<uint32_t> rows_cpu;
    std::vector<uint32_t> cols_cpu;

    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, matrix_res_cpu);

    uint32_t size = rows_cpu.size();
    for (uint32_t i = 0; i < size; ++i) {
        if (matrix_res_gpu.rows_indices_cpu()[i] != rows_cpu[i] ||
            matrix_res_gpu.cols_indices_cpu()[i] != cols_cpu[i]) {
            std::cerr << "incorrect kronecker" << std::endl;
        }
    }

    std::cout << "correct kronecker" << std::endl;


}

int main() {
//    testBitonicSort();
//    testReduceDuplicates();
    testMatrixAddition();
//    testKronecker();
}


