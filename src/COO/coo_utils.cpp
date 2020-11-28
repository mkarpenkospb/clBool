#include <cstdint>
#include "coo_utils.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../fast_random.h"
#include <vector>
#include <iostream>
#include <algorithm>

namespace coo_utils {


    void fill_random_matrix(std::vector<uint32_t> &rows, std::vector<uint32_t> &cols, uint32_t max_size) {
        uint32_t n = rows.size();
        FastRandom r(n);
        for (uint32_t i = 0; i < n; ++i) {
            rows[i] = r.next() % max_size;
            cols[i] = r.next() % max_size;
        }
    }

    void
    form_cpu_matrix(matrix_cpp_cpu &matrix_out, const std::vector<uint32_t> &rows, const std::vector<uint32_t> &cols) {
        matrix_out.resize(rows.size());
        std::transform(rows.begin(), rows.end(), cols.begin(), matrix_out.begin(),
                       [](uint32_t i, uint32_t j) -> coordinates { return {i, j}; });

    }

    void get_vectors_from_cpu_matrix(std::vector<uint32_t> &rows_out, std::vector<uint32_t> &cols_out,
                                     const matrix_cpp_cpu &matrix) {
        uint32_t n = matrix.size();

        rows_out.resize(matrix.size());
        cols_out.resize(matrix.size());

        for (uint32_t i = 0; i < n; ++i) {
            rows_out[i] = matrix[i].first;
            cols_out[i] = matrix[i].second;
        }

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

    matrix_cpp_cpu generate_random_matrix_cpu(uint32_t pseudo_size, uint32_t max_size) {

        std::vector<uint32_t> rows(pseudo_size);
        std::vector<uint32_t> cols(pseudo_size);

        fill_random_matrix(rows, cols, max_size);

        matrix_cpp_cpu m_cpu;
        form_cpu_matrix(m_cpu, rows, cols);
        std::sort(m_cpu.begin(), m_cpu.end());

        m_cpu.erase(std::unique(m_cpu.begin(), m_cpu.end()), m_cpu.end());

        return m_cpu;
    }

    matrix_coo matrix_coo_from_cpu(Controls &controls, const matrix_cpp_cpu &m_cpu) {
        std::vector<uint32_t> rows;
        std::vector<uint32_t> cols;

        get_vectors_from_cpu_matrix(rows, cols, m_cpu);

        uint32_t n_rows = *std::max_element(rows.begin(), rows.end());
        uint32_t n_cols = *std::max_element(cols.begin(), cols.end());
        uint32_t nnz = m_cpu.size();

        return matrix_coo(controls, n_rows, n_cols, nnz, std::move(rows), std::move(cols), true);
    }

    void
    matrix_addition_cpu(matrix_cpp_cpu &matrix_out, const matrix_cpp_cpu &matrix_a, const matrix_cpp_cpu &matrix_b) {

        std::merge(matrix_a.begin(), matrix_a.end(), matrix_b.begin(), matrix_b.end(),
                   std::back_inserter(matrix_out));

        matrix_out.erase(std::unique(matrix_out.begin(), matrix_out.end()), matrix_out.end());

    }

    void
    kronecker_product_cpu(matrix_cpp_cpu &matrix_out, const matrix_cpp_cpu &matrix_a, const matrix_cpp_cpu &matrix_b) {
        auto less_for_rows = [](const coordinates &a, const coordinates &b) -> bool {
            return a.first < b.first;
        };
        auto less_for_cols = [](const coordinates &a, const coordinates &b) -> bool {
            return a.second < b.second;
        };

        uint32_t matrix_b_nRows = std::max_element(matrix_b.begin(), matrix_b.end(), less_for_rows)->first;
        uint32_t matrix_b_nCols = std::max_element(matrix_b.begin(), matrix_b.end(), less_for_cols)->second;

        for (const auto &coord_a: matrix_a) {
            for (const auto &coord_b: matrix_b) {
                matrix_out.emplace_back(coord_a.first * matrix_b_nRows + coord_b.first,
                                     coord_a.second * matrix_b_nCols + coord_b.second);
            }
        }
        std::sort(matrix_out.begin(), matrix_out.end());
    }
}