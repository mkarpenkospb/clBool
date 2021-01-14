#pragma once

#include <vector>
#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"

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

    matrix_coo_cpu generate_random_matrix_cpu(uint32_t pseudo_nnz, uint32_t max_size = 1024);

    matrix_coo matrix_coo_from_cpu(Controls &controls, const matrix_coo_cpu &m_cpu);

    void
    matrix_addition_cpu(matrix_coo_cpu &matrix_out, const matrix_coo_cpu &matrix_a, const matrix_coo_cpu &matrix_b);

    void
    kronecker_product_cpu(matrix_coo_cpu &matrix_out, const matrix_coo_cpu &matrix_a, const matrix_coo_cpu &matrix_b);

    void print_matrix(const matrix_coo_cpu& m_cpu);

    void get_rows_pointers_and_compressed(cpu_buffer &rows_pointers,
                                          cpu_buffer &rows_compressed,
                                          const matrix_coo_cpu &matrix_cpu);

    void get_workload (cpu_buffer &workload,
                       const matrix_coo_cpu &matrix_a_cpu,
                       const cpu_buffer &a_rows_pointers,
                       const cpu_buffer &b_rows_pointers,
                       const cpu_buffer &b_rows_compressed,
                       uint32_t a_nzr
    );


    // cpu class for double compressed matrix

    class matrix_dcsr_cpu {
        cpu_buffer rows_pointers;
        cpu_buffer rows_compressed;
        cpu_buffer cols_indices;

    public:
        matrix_dcsr_cpu(cpu_buffer rows_pointers, cpu_buffer rows_compressed, cpu_buffer cpu_indices)
        : rows_pointers(std::move(rows_pointers))
        , rows_compressed(std::move(rows_compressed))
        , cols_indices(std::move(cpu_indices))
        {}

        matrix_dcsr_cpu() = default;

        matrix_dcsr_cpu& operator=(matrix_dcsr_cpu other) {
            rows_pointers = std::move(other.rows_pointers);
            rows_compressed = std::move(other.rows_compressed);
            cols_indices = std::move(other.cols_indices);
        }

        cpu_buffer& get_rows_pointers() {
            return rows_pointers;
        }
        cpu_buffer& get_rows_compressed() {
            return rows_compressed;
        }
        cpu_buffer& get_cols_indices() {
            return cols_indices;
        }

        const cpu_buffer& get_rows_pointers() const {
            return rows_pointers;
        }
        const cpu_buffer& get_rows_compressed() const {
            return rows_compressed;
        }
        const cpu_buffer& get_cols_indices() const {
            return cols_indices;
        }

    };

    void matrix_multiplication_cpu(matrix_dcsr_cpu &c, const matrix_coo_cpu &a, const matrix_coo_cpu &b);
}
