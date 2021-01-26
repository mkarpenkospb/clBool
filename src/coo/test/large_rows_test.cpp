#include "coo_tests.hpp"


#include "coo_tests.hpp"
#include "../../cl_includes.hpp"
#include "../../library_classes/controls.hpp"
#include "../../library_classes/matrix_dcsr.hpp"
#include "../../utils.hpp"
#include "../coo_utils.hpp"
#include "../dscr_matrix_multiplication.hpp"

using namespace coo_utils;
using namespace utils;

namespace {
    void compare_matrices(Controls &controls, matrix_dcsr m_gpu, matrix_dcsr_cpu m_cpu) {
        compare_buffers(controls, m_gpu.rows_pointers_gpu(), m_cpu.rows_pointers(), m_gpu.nzr() + 1);
        compare_buffers(controls, m_gpu.rows_compressed_gpu(), m_cpu.rows_compressed(), m_gpu.nzr());
        compare_buffers(controls, m_gpu.cols_indices_gpu(), m_cpu.cols_indices(), m_gpu.nnz());
    }
}


const uint32_t BINS_NUM = 38;

void largeRowsTest() {
    Controls controls = utils::create_controls();
    uint32_t max_size = 1000;
    auto generated = generate_random_matrices_large(max_size, 342787282);

    matrix_dcsr_cpu a_cpu = generated.first;
    matrix_dcsr_cpu b_cpu = generated.second;
    matrix_dcsr_cpu c_cpu;

    matrix_multiplication_cpu(c_cpu, a_cpu, b_cpu);
    std::cout << "matrix_multiplication_cpu finished" << std::endl;
//    print_matrix(c_cpu);

    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);
    matrix_dcsr c_gpu;

    matrix_multiplication(controls, c_gpu, a_gpu, b_gpu);

//    print_matrix(controls, c_gpu);

    compare_matrices(controls, c_gpu, c_cpu);

//    print_matrix(c_cpu);

}

