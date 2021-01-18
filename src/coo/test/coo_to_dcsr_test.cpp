#include <algorithm>
#include "coo_tests.hpp"
#include "../../cl_includes.hpp"
#include "../../library_classes/controls.hpp"
#include "../../utils.hpp"
#include "../coo_utils.hpp"
#include "../coo_matrix_multiplication.hpp"


using coo_utils::matrix_coo_cpu;
using cpu_buffer = std::vector<uint32_t>;


void testCOOtoDCSR() {
    Controls controls = utils::create_controls();
    // ----------------------------------------- create matrices ----------------------------------------

    uint32_t size = 1000463;
    uint32_t max_size = 100024;
    matrix_coo_cpu m_cpu = coo_utils::generate_random_matrix_coo_cpu(size, max_size);
    cpu_buffer rows_pointers_cpu;
    cpu_buffer rows_compressed_cpu;
    coo_utils::get_rows_pointers_and_compressed(rows_pointers_cpu, rows_compressed_cpu, m_cpu);
//    coo_utils::print_matrix(m_cpu);

    matrix_coo matrix_gpu = coo_utils::matrix_coo_from_cpu(controls, m_cpu);


    cl::Buffer rows_pointers;
    cl::Buffer rows_compressed;
    uint32_t a_nzr;

    create_rows_pointers(controls, rows_pointers, rows_compressed, matrix_gpu.rows_indices_gpu(),
                         matrix_gpu.nnz(), a_nzr);
//    utils::print_gpu_buffer(controls, rows_pointers, 10);
//    utils::print_gpu_buffer(controls, _rows_compressed, 10);

    utils::compare_buffers(controls, rows_pointers, rows_pointers_cpu, rows_pointers_cpu.size());
    utils::compare_buffers(controls, rows_compressed, rows_compressed_cpu, rows_compressed_cpu.size());
}