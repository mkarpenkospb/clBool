#include "coo_tests.hpp"


#include "coo_tests.hpp"
#include "../../cl_includes.hpp"
#include "../../library_classes/controls.hpp"
#include "../../utils.hpp"
#include "../coo_utils.hpp"
#include "../dscr_matrix_multiplication.hpp"

using namespace coo_utils;
using namespace utils;
const uint32_t BINS_NUM = 38;
namespace {
    void compare_matrices(Controls &controls, matrix_dcsr m_gpu, matrix_dcsr_cpu m_cpu) {
        compare_buffers(controls, m_gpu.rows_pointers_gpu(), m_cpu.rows_pointers(), m_gpu.nzr() + 1);
        compare_buffers(controls, m_gpu.rows_compressed_gpu(), m_cpu.rows_compressed(), m_gpu.nzr());
        compare_buffers(controls, m_gpu.cols_indices_gpu(), m_cpu.cols_indices(), m_gpu.nnz());
    }
}


void testESC() {
    Controls controls = utils::create_controls();
    uint32_t max_size = 400;
    for (int i = 234523; i  < 234523 + 20; ++i) {
        auto generated = generate_random_matrices_esc(max_size + (i % 234523), i);
        matrix_dcsr_cpu a_cpu = generated.first;
        matrix_dcsr_cpu b_cpu = generated.second;
        matrix_dcsr_cpu c_cpu;
//
//    printf("a_cpu: \n");
//    print_matrix(a_cpu);
//    printf("b_cpu: \n");
//    print_matrix(b_cpu);

        matrix_multiplication_cpu(c_cpu, a_cpu, b_cpu);
//    printf("c_cpu: \n");
//    print_matrix(c_cpu);

        matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
        matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
        matrix_dcsr c_gpu;
//    print_matrix(controls, a_gpu);
//    print_matrix(controls, b_gpu);
        matrix_multiplication(controls, c_gpu, a_gpu, b_gpu);
        compare_matrices(controls, c_gpu, c_cpu);
    }
}




