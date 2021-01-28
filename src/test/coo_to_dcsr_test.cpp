#include <algorithm>
#include "coo_tests.hpp"
#include "../coo/coo_utils.hpp"
#include "../dcsr/dscr_matrix_multiplication.hpp"


void testCOOtoDCSR() {
    Controls controls = utils::create_controls();
    // ----------------------------------------- create matrices ----------------------------------------

    uint32_t size = 1000463;
    uint32_t max_size = 100024;
    matrix_coo_cpu m_cpu = coo_utils::generate_random_matrix_coo_cpu(size, max_size);
    cpu_buffer rows_pointers_cpu;
    cpu_buffer rows_compressed_cpu;
    coo_utils::get_rows_pointers_and_compressed(rows_pointers_cpu, rows_compressed_cpu, m_cpu);

    matrix_coo matrix_gpu = coo_utils::matrix_coo_from_cpu(controls, m_cpu);

    matrix_dcsr m_dcsr = coo_to_dcsr_gpu(controls, matrix_gpu);

    utils::compare_buffers(controls, m_dcsr.rows_pointers_gpu(), rows_pointers_cpu, rows_pointers_cpu.size());
    utils::compare_buffers(controls, m_dcsr.rows_compressed_gpu(), rows_compressed_cpu, rows_compressed_cpu.size());


    matrix_coo another_one = dcsr_to_coo(controls, m_dcsr);


}