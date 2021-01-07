#pragma once

#include <vector>
#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"

namespace coo_utils {
    using coordinates = std::pair<uint32_t, uint32_t>;
    using matrix_cpp_cpu = std::vector<coordinates>;

    void check_correctness(const std::vector<uint32_t> &rows, const std::vector<uint32_t> &cols);

    void fill_random_matrix(std::vector<uint32_t> &rows, std::vector<uint32_t> &cols, uint32_t max_size = 1024);

    void
    form_cpu_matrix(matrix_cpp_cpu &matrix_out, const std::vector<uint32_t> &rows, const std::vector<uint32_t> &cols);

    void get_vectors_from_cpu_matrix(std::vector<uint32_t> &rows_out, std::vector<uint32_t> &cols_out,
                                     const matrix_cpp_cpu &matrix);

    matrix_cpp_cpu generate_random_matrix_cpu(uint32_t pseudo_nnz, uint32_t max_size = 1024);

    matrix_coo matrix_coo_from_cpu(Controls &controls, const matrix_cpp_cpu &m_cpu);

    void
    matrix_addition_cpu(matrix_cpp_cpu &matrix_out, const matrix_cpp_cpu &matrix_a, const matrix_cpp_cpu &matrix_b);

    void
    kronecker_product_cpu(matrix_cpp_cpu &matrix_out, const matrix_cpp_cpu &matrix_a, const matrix_cpp_cpu &matrix_b);
}
