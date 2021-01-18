#include <cstdint>
#include "coo_utils.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../fast_random.h"
#include "../library_classes/matrix_dcsr.hpp"
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

    matrix_coo_cpu generate_random_matrix_coo_cpu(uint32_t pseudo_nnz, uint32_t max_size) {

        cpu_buffer rows(pseudo_nnz);
        cpu_buffer cols(pseudo_nnz);

        fill_random_matrix(rows, cols, max_size);

        matrix_coo_cpu m_cpu;
        form_cpu_matrix(m_cpu, rows, cols);
        std::sort(m_cpu.begin(), m_cpu.end());

        m_cpu.erase(std::unique(m_cpu.begin(), m_cpu.end()), m_cpu.end());

        return m_cpu;
    }

    matrix_dcsr_cpu coo_to_dcsr_cpu(const matrix_coo_cpu &matrix_coo) {


        cpu_buffer rows_pointers;
        cpu_buffer rows_compressed;
        cpu_buffer cols_indices;


        size_t position = 0;
        uint32_t curr_row = matrix_coo.front().first;
        rows_compressed.push_back(curr_row);
        rows_pointers.push_back(position);

        for (const auto &item: matrix_coo) {
            cols_indices.push_back(item.second);
            if (item.first != curr_row) {
                curr_row = item.first;
                rows_compressed.push_back(curr_row);
                rows_pointers.push_back(position);
            }
            position ++;
        }
        rows_pointers.push_back(position);

        return matrix_dcsr_cpu(rows_pointers, rows_compressed, cols_indices);
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

        if (m_cpu.cols_indices().empty()) {
            std::cout << "empty matrix" << std::endl;
            return;
        }

        uint32_t m_cpu_nzr = m_cpu.rows_compressed().size();
        for (uint32_t i = 0; i < m_cpu_nzr; ++i) {
            std::cout << "row " << m_cpu.rows_compressed()[i] << ": ";
            uint32_t start = m_cpu.rows_pointers()[i];
            uint32_t end = m_cpu.rows_pointers()[i + 1];

            for (uint32_t j = start; j < end; ++j) {
                std::cout << m_cpu.cols_indices()[j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void print_matrix(const matrix_dcsr& m_cpu) {
////        cpu_buffer
//
//
//
//
//
//        if (m_cpu.cols_indices().empty()) {
//            std::cout << "empty matrix" << std::endl;
//            return;
//        }
//
//        uint32_t m_cpu_nzr = m_cpu.rows_compressed().size();
//        for (uint32_t i = 0; i < m_cpu_nzr; ++i) {
//            std::cout << "row " << m_cpu.rows_compressed()[i] << ": ";
//            uint32_t start = m_cpu.rows_pointers()[i];
//            uint32_t end = m_cpu.rows_pointers()[i + 1];
//
//            for (uint32_t j = start; j < end; ++j) {
//                std::cout << m_cpu.cols_indices()[j] << ", ";
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
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
                      const matrix_dcsr_cpu &a,
                      const matrix_dcsr_cpu &b
                      ) {

        for (uint32_t i = 0; i < a.rows_compressed().size(); ++i) {
            workload[i] = 0;
            uint32_t start = a.rows_pointers()[i];
            uint32_t end = a.rows_pointers()[i + 1];
            for (uint32_t j = start; j < end; ++j) {
                auto it = std::find(b.rows_compressed().begin(), b.rows_compressed().end(), a.cols_indices()[j]);
                if (it != b.rows_compressed().end()) {
                    uint32_t pos = it - b.rows_compressed().begin();
                    workload[i] += b.rows_pointers()[pos + 1] - b.rows_pointers()[pos];
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

    void matrix_multiplication_cpu(matrix_dcsr_cpu &c,
                                   const matrix_dcsr_cpu &a,
                                   const matrix_dcsr_cpu &b) {

        uint32_t a_nzr = a.rows_compressed().size();
        cpu_buffer c_cols_indices;
        cpu_buffer c_rows_pointers;
        cpu_buffer c_rows_compressed;

        uint32_t current_pointer = 0;
        cpu_buffer current_row;
        for (uint32_t i = 0; i < a_nzr; ++i) {
            uint32_t start = a.rows_pointers()[i];
            uint32_t end = a.rows_pointers()[i + 1];
            bool is_row = false;

            for (uint32_t j = start; j < end; ++j) {
                auto it = std::find(b.rows_compressed().begin(), b.rows_compressed().end(), a.cols_indices()[j]);
                if (it != b.rows_compressed().end()) {
                    if (!is_row) {
                        c_rows_pointers.push_back(current_pointer);
                        c_rows_compressed.push_back(a.rows_compressed()[i]);
                        is_row = true;
                    }
                    uint32_t pos = it - b.rows_compressed().begin();
                    uint32_t b_start = b.rows_pointers()[pos];
                    uint32_t b_end = b.rows_pointers()[pos + 1];
                    for (uint32_t k = b_start; k < b_end; ++k) {
                        current_row.push_back(b.cols_indices()[k]);
                    }
                }
            }
            if (current_row.empty()) continue;

            std::sort(current_row.begin(), current_row.end());
            current_row.erase(std::unique(current_row.begin(), current_row.end()), current_row.end());
            c_cols_indices.insert(c_cols_indices.end(), current_row.begin(), current_row.end());

            current_pointer += current_row.size();
            current_row = cpu_buffer();
        }

        c_rows_pointers.push_back(current_pointer);
        c = matrix_dcsr_cpu(std::move(c_rows_pointers), std::move(c_rows_compressed), std::move(c_cols_indices));
    }


    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, coo_utils::matrix_dcsr_cpu &m, uint32_t size) {

        cl::Buffer rows_pointers(controls.context, m.rows_pointers().begin(), m.rows_pointers().end(), false);
        cl::Buffer rows_compressed(controls.context, m.rows_compressed().begin(), m.rows_compressed().end(), false);
        cl::Buffer cols_indices(controls.context, m.cols_indices().begin(), m.cols_indices().end(), false);

        return matrix_dcsr(rows_pointers, rows_compressed, cols_indices,
                           size, size, m.cols_indices().size(), m.rows_compressed().size());

    }
}