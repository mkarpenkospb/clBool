#include <cstdint>
#include "coo_utils.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../fast_random.h"
#include <vector>
#include <iostream>
#include <algorithm>

namespace coo_utils {
    using cpu_buffer = std::vector<uint32_t>;

    void fill_random_matrix(cpu_buffer &rows, cpu_buffer &cols, uint32_t max_size) {
        uint32_t n = rows.size();
        FastRandom r(n);
        for (uint32_t i = 0; i < n; ++i) {
            rows[i] = r.next() % max_size;
            cols[i] = r.next() % max_size;
        }
    }

    void
    form_cpu_matrix(matrix_coo_cpu &matrix_out, const cpu_buffer &rows, const cpu_buffer &cols) {
        matrix_out.resize(rows.size());
        std::transform(rows.begin(), rows.end(), cols.begin(), matrix_out.begin(),
                       [](uint32_t i, uint32_t j) -> coordinates { return {i, j}; });

    }

    void get_vectors_from_cpu_matrix(cpu_buffer &rows_out, cpu_buffer &cols_out,
                                     const matrix_coo_cpu &matrix) {
        uint32_t n = matrix.size();

        rows_out.resize(matrix.size());
        cols_out.resize(matrix.size());

        for (uint32_t i = 0; i < n; ++i) {
            rows_out[i] = matrix[i].first;
            cols_out[i] = matrix[i].second;
        }

    }


    void check_correctness(const cpu_buffer &rows, const cpu_buffer &cols) {
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

    matrix_coo_cpu generate_random_matrix_cpu(uint32_t pseudo_nnz, uint32_t max_size) {

        std::vector<uint32_t> rows(pseudo_nnz);
        std::vector<uint32_t> cols(pseudo_nnz);

        fill_random_matrix(rows, cols, max_size);

        matrix_coo_cpu m_cpu;
        form_cpu_matrix(m_cpu, rows, cols);
        std::sort(m_cpu.begin(), m_cpu.end());

        m_cpu.erase(std::unique(m_cpu.begin(), m_cpu.end()), m_cpu.end());

        return m_cpu;
    }

    matrix_coo matrix_coo_from_cpu(Controls &controls, const matrix_coo_cpu &m_cpu) {
        cpu_buffer rows;
        cpu_buffer cols;

        get_vectors_from_cpu_matrix(rows, cols, m_cpu);

        uint32_t n_rows = *std::max_element(rows.begin(), rows.end());
        uint32_t n_cols = *std::max_element(cols.begin(), cols.end());
        uint32_t nnz = m_cpu.size();

        return matrix_coo(controls, n_rows, n_cols, nnz, std::move(rows), std::move(cols), true);
    }

    void
    matrix_addition_cpu(matrix_coo_cpu &matrix_out, const matrix_coo_cpu &matrix_a, const matrix_coo_cpu &matrix_b) {

        std::merge(matrix_a.begin(), matrix_a.end(), matrix_b.begin(), matrix_b.end(),
                   std::back_inserter(matrix_out));

        matrix_out.erase(std::unique(matrix_out.begin(), matrix_out.end()), matrix_out.end());

    }

    void
    kronecker_product_cpu(matrix_coo_cpu &matrix_out, const matrix_coo_cpu &matrix_a, const matrix_coo_cpu &matrix_b) {
        auto less_for_rows = [](const coordinates &a, const coordinates &b) -> bool {
            return a.first < b.first;
        };
        auto less_for_cols = [](const coordinates &a, const coordinates &b) -> bool {
            return a.second < b.second;
        };

        uint32_t matrix_b_nRows = std::max_element(matrix_b.begin(), matrix_b.end(), less_for_rows)->first;
        uint32_t matrix_b_nCols = std::max_element(matrix_b.begin(), matrix_b.end(), less_for_cols)->second;

        matrix_out.resize(matrix_a.size() * matrix_b.size());

        uint32_t i = 0;
        for (const auto &coord_a: matrix_a) {
            for (const auto &coord_b: matrix_b) {
                matrix_out[i] = coordinates(coord_a.first * matrix_b_nRows + coord_b.first,
                                            coord_a.second * matrix_b_nCols + coord_b.second);
                ++i;
            }
        }
        std::sort(matrix_out.begin(), matrix_out.end());
    }


    void print_matrix(const matrix_coo_cpu& m_cpu) {
        if (m_cpu.empty()) {
            std::cout << "empty matrix" << std::endl;
            return;
        }

        uint32_t curr_row = m_cpu.front().first;
        std::cout << "row " << curr_row << ": ";
        for (const auto &item: m_cpu) {
            if (item.first != curr_row) {
                curr_row = item.first;
                std::cout << std::endl;
                std::cout << "row " << curr_row << ": ";
            }
            std::cout << item.second << ", ";
        }
        std::cout << std::endl;
    }


    void print_matrix(const matrix_dcsr_cpu& m_cpu) {
        std::cout << std::endl;

        if (m_cpu.get_cols_indices().empty()) {
            std::cout << "empty matrix" << std::endl;
            return;
        }

        uint32_t m_cpu_nzr = m_cpu.get_rows_compressed().size();
        for (uint32_t i = 0; i < m_cpu_nzr; ++i) {
            std::cout << "row " << m_cpu.get_rows_compressed()[i] << ": ";
            uint32_t start = m_cpu.get_rows_pointers()[i];
            uint32_t end = m_cpu.get_rows_pointers()[i + 1];

            for (uint32_t j = start; j < end; ++j) {
                std::cout << m_cpu.get_cols_indices()[j] << ", ";
            }
        }
        std::cout << std::endl;
    }

    void get_rows_pointers_and_compressed(cpu_buffer &rows_pointers,
                                          cpu_buffer &rows_compressed,
                                          const matrix_coo_cpu &matrix_cpu) {
        if (matrix_cpu.empty()) return;

        size_t position = 0;
        uint32_t curr_row = matrix_cpu.front().first;
        rows_compressed.push_back(curr_row);
        rows_pointers.push_back(position);
        for (const auto &item: matrix_cpu) {
            if (item.first != curr_row) {
                curr_row = item.first;
                rows_compressed.push_back(curr_row);
                rows_pointers.push_back(position);
            }
            position ++;
        }
        rows_pointers.push_back(position);
        std::cout << std::endl;

    }

    void get_workload (cpu_buffer &workload,
                      const matrix_coo_cpu &matrix_a_cpu,
                      const cpu_buffer &a_rows_pointers,
                      const cpu_buffer &b_rows_pointers,
                      const cpu_buffer &b_rows_compressed,
                      uint32_t a_nzr
                      ) {

        for (uint32_t i = 0; i < a_nzr; ++i) {
            workload[i] = 0;
            uint32_t start = a_rows_pointers[i];
            uint32_t end = a_rows_pointers[i + 1];
            for (uint32_t j = start; j < end; ++j) {
                auto it = std::find(b_rows_compressed.begin(), b_rows_compressed.end(), matrix_a_cpu[j].second);
                if (it != b_rows_compressed.end()) {
                    uint32_t pos = it - b_rows_compressed.begin();
                    workload[i] += b_rows_pointers[pos + 1] - b_rows_pointers[pos];
                }
            }
        }
    }




//    void matrix_coo_to_dcsr_cpu(matrix_dcsr_cpu &out, const matrix_coo_cpu &in) {
//
//    }


    /*
     * штука нужна для сравнения с gpu, поэтому возвращать будем своего рода matrix_dcsr_cpu
     */

    void matrix_multiplication_cpu(matrix_dcsr_cpu &c, const matrix_coo_cpu &a, const matrix_coo_cpu &b) {
        cpu_buffer a_rows_pointers;
        cpu_buffer a_rows_compressed;
        get_rows_pointers_and_compressed(a_rows_pointers, a_rows_compressed, a);

        cpu_buffer b_rows_pointers;
        cpu_buffer b_rows_compressed;
        get_rows_pointers_and_compressed(b_rows_pointers, b_rows_compressed, b);

        // практически копируем код get_workload, чтобы создать матрицу.
        // напихаем ряды в один вектор и потом отсортируем. Или просто merg-ы сделаем.
        // можно две версии написать
        uint32_t a_nzr = a_rows_compressed.size();
        cpu_buffer c_cols_indices;
        cpu_buffer c_rows_pointers;
        cpu_buffer c_rows_compressed;

        uint32_t current_pointer = 0;
        cpu_buffer current_row;
        for (uint32_t i = 0; i < a_nzr; ++i) {
            uint32_t start = a_rows_pointers[i];
            uint32_t end = a_rows_pointers[i + 1];
            bool is_row = false;

            for (uint32_t j = start; j < end; ++j) {
                auto it = std::find(b_rows_compressed.begin(), b_rows_compressed.end(), a_rows_compressed[j]);
                if (it != b_rows_compressed.end()) {
                    if (!is_row) {
                        c_rows_pointers.push_back(current_pointer);
                        c_rows_compressed.push_back(a_rows_compressed[i]);
                    }
                    is_row = true;
                    uint32_t pos = it - b_rows_compressed.begin();
                    uint32_t b_start = b_rows_pointers[pos];
                    uint32_t b_end = b_rows_pointers[pos + 1];
                    for (uint32_t k = b_start; k < b_end; ++k) {
                        current_row.push_back(b[k].second);
                    }
                }
            }
            if (current_row.empty()) continue;

            std::sort(current_row.begin(), current_row.end());
            current_row.erase(std::unique(current_row.begin(), current_row.end()), current_row.end());
            c_rows_compressed.insert(c_rows_compressed.begin(), current_row.begin(), current_row.end());

            current_pointer += current_row.size();
            current_row = cpu_buffer();
        }

        c_rows_pointers.push_back(current_pointer);
        c = matrix_dcsr_cpu(std::move(c_rows_pointers), std::move(c_rows_compressed), std::move(c_cols_indices));
    }

}