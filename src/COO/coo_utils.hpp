#pragma once
#include <vector>

namespace coo_utils{
    using coordinates = std::pair<uint32_t, uint32_t>;
    using matrix_cpp_cpu = std::vector<coordinates>;
    void check_correctness(const std::vector<uint32_t> &rows, const std::vector<uint32_t> &cols);
    void fill_random_matrix(std::vector<uint32_t>& rows, std::vector<uint32_t>& cols);
    void form_cpu_matrix(matrix_cpp_cpu& matrix_out, std::vector<uint32_t>& rows, std::vector<uint32_t>& cols);
    void get_vectors_from_cpu_matrix(std::vector<uint32_t>& rows_out, std::vector<uint32_t>& cols_out, matrix_cpp_cpu& matrix);
}
