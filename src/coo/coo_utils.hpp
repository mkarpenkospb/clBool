#pragma once

#include <vector>
#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_dcsr.hpp"

namespace coo_utils {
    using coordinates = std::pair<uint32_t, uint32_t>;
    using matrix_coo_cpu = std::vector<coordinates>;
    using cpu_buffer = std::vector<uint32_t>;

    void check_correctness(const cpu_buffer &rows, const cpu_buffer &cols);

    void fill_random_matrix(cpu_buffer &rows, cpu_buffer &cols, uint32_t max_size = 1024);

    void
    form_cpu_matrix(matrix_coo_cpu &matrix_out, const cpu_buffer &rows, const cpu_buffer &cols);

    void get_vectors_from_cpu_matrix(cpu_buffer &rows_out, cpu_buffer &cols_out,
                                     const matrix_coo_cpu &matrix);

    matrix_coo_cpu generate_random_matrix_coo_cpu(uint32_t pseudo_nnz, uint32_t max_size = 1024);

    matrix_coo matrix_coo_from_cpu(Controls &controls, const matrix_coo_cpu &m_cpu);

    void
    matrix_addition_cpu(matrix_coo_cpu &matrix_out, const matrix_coo_cpu &matrix_a, const matrix_coo_cpu &matrix_b);

    void
    kronecker_product_cpu(matrix_coo_cpu &matrix_out, const matrix_coo_cpu &matrix_a, const matrix_coo_cpu &matrix_b);

    void print_matrix(const matrix_coo_cpu &m_cpu);

    void get_rows_pointers_and_compressed(cpu_buffer &rows_pointers,
                                          cpu_buffer &rows_compressed,
                                          const matrix_coo_cpu &matrix_cpu);



    // cpu class for double compressed matrix

    class matrix_dcsr_cpu {
        cpu_buffer _rows_pointers;
        cpu_buffer _rows_compressed;
        cpu_buffer _cols_indices;

    public:
        matrix_dcsr_cpu(cpu_buffer rows_pointers, cpu_buffer rows_compressed, cpu_buffer cpu_indices)
                : _rows_pointers(std::move(rows_pointers)), _rows_compressed(std::move(rows_compressed)),
                  _cols_indices(std::move(cpu_indices)) {}

        matrix_dcsr_cpu() = default;

        matrix_dcsr_cpu &operator=(matrix_dcsr_cpu other) {
            _rows_pointers = std::move(other._rows_pointers);
            _rows_compressed = std::move(other._rows_compressed);
            _cols_indices = std::move(other._cols_indices);
            return *this;
        }

        cpu_buffer &rows_pointers() {
            return _rows_pointers;
        }

        cpu_buffer &rows_compressed() {
            return _rows_compressed;
        }

        cpu_buffer &cols_indices() {
            return _cols_indices;
        }

        const cpu_buffer &rows_pointers() const {
            return _rows_pointers;
        }

        const cpu_buffer &rows_compressed() const {
            return _rows_compressed;
        }

        const cpu_buffer &cols_indices() const {
            return _cols_indices;
        }

    };

    void matrix_multiplication_cpu(matrix_dcsr_cpu &c,
                                   const matrix_dcsr_cpu &a,
                                   const matrix_dcsr_cpu &b);

    matrix_dcsr_cpu coo_to_dcsr_cpu(const matrix_coo_cpu &matrix_coo);

    matrix_dcsr matrix_dcsr_from_cpu(Controls &controls, coo_utils::matrix_dcsr_cpu &m, uint32_t size);

    void get_workload(cpu_buffer &workload,
                      const matrix_dcsr_cpu &a_cpu,
                      const matrix_dcsr_cpu &b_cpu);


    std::pair<matrix_dcsr_cpu, matrix_dcsr_cpu> generate_random_matrices_esc(uint32_t max_size, uint32_t seed);
    std::pair<matrix_dcsr_cpu, matrix_dcsr_cpu> generate_random_matrices_large(uint32_t max_size, uint32_t seed);


    void print_matrix(const matrix_dcsr_cpu &m_cpu, uint32_t index = -1);
    void print_matrix(Controls &controls, const matrix_dcsr& m_gpu, uint32_t index = -1);
}
