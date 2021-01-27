#include <cmath>
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
//        if (m_gpu.nnz() != m_cpu.cols_indices().size()) {
//            std::cout << "diff nnz, gpu: " << m_gpu.nnz() << " vs cpu: " << m_cpu.cols_indices().size() << std::endl;
//        }
        compare_buffers(controls, m_gpu.rows_pointers_gpu(), m_cpu.rows_pointers(), m_gpu.nzr() + 1);
        compare_buffers(controls, m_gpu.rows_compressed_gpu(), m_cpu.rows_compressed(), m_gpu.nzr());
        compare_buffers(controls, m_gpu.cols_indices_gpu(), m_cpu.cols_indices(), m_gpu.nnz());
    }
}

const uint32_t BINS_NUM = 38;

void test_multiplication() {
    Controls controls = utils::create_controls();
//    for (uint32_t i = 1; i < 10; i ++) {
//        std::cout << "i = " << i << std::endl;
        uint32_t max_size = 2035;
        uint32_t nnz_max = std::max(10u, max_size * 15);

        matrix_dcsr_cpu a_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_max, max_size));
//        print_cpu_buffer()
//        matrix_dcsr_cpu b_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_max, max_size - 5));
        matrix_dcsr_cpu c_cpu;
//        print_matrix(a_cpu);
        matrix_multiplication_cpu(c_cpu, a_cpu, a_cpu);

        std::cout << "matrix_multiplication_cpu finished" << std::endl;

        matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
        matrix_dcsr b_gpu = a_gpu;
        matrix_dcsr c_gpu;

//        print_matrix(c_cpu, 1617);
        std::cout << "s\n";
        matrix_multiplication(controls, c_gpu, a_gpu, b_gpu);
        std::cout << "e\n";
//        print_matrix(controls, c_gpu, 1044);
        compare_matrices(controls, c_gpu, c_cpu);
//    }
}

