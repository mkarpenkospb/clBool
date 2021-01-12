#include <algorithm>
#include "../../cl_includes.hpp"
#include "coo_tests.hpp"
#include "../../library_classes/controls.hpp"
#include "../coo_utils.hpp"
#include "../coo_kronecker_product.hpp"


using coo_utils::matrix_coo_cpu;

void testKronecker() {
    Controls controls = utils::create_controls();

    matrix_coo_cpu matrix_res_cpu;
    // first argument is nnz (nnz before reducing duplicates after random)
    // second is the maximum possible matrix size
    matrix_coo_cpu matrix_a_cpu = coo_utils::generate_random_matrix_cpu(10000, 3342);
    matrix_coo_cpu matrix_b_cpu = coo_utils::generate_random_matrix_cpu(10000, 2234);

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