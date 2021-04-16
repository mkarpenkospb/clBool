#include "coo_tests.hpp"

#include <random>
#include "../dcsr/dcsr.hpp"
#include <matrices_conversions.hpp>
#include "../coo/coo_utils.hpp"

using namespace coo_utils;
using namespace utils;

void test_transpose() {
    Controls controls = create_controls();
    timer t;
    for (uint32_t k = 10; k < 30; ++k) {
        for (int size = 20; size < 400; size += 200) {
            uint32_t max_size = size;
            uint32_t nnz_max = std::max(10u, max_size * k);

            if (DEBUG_ENABLE)
                Log() << " ------------------------------- k = " << k << ", size = " << size
                      << " -------------------------------------------\n"
                      << "max_size = " << size << ", nnz_max = " << nnz_max;

            matrix_coo_cpu a_coo_cpu = generate_coo_cpu(nnz_max, max_size);
            matrix_dcsr_cpu a_dcsr_cpu = coo_to_dcsr_cpu(a_coo_cpu);
            a_coo_cpu.transpose();
            matrix_dcsr_cpu a_dcsr_cpu_tr = coo_to_dcsr_cpu(a_coo_cpu);

            matrix_dcsr a_gpu;
            t.restart(); {

                a_gpu = matrix_dcsr_from_cpu(controls, a_dcsr_cpu, max_size);

            } t.elapsed();
            if (DEBUG_ENABLE) Log() << "matrix_dcsr_from_cpu in " << t.last_elapsed();

            t.restart(); {

                transpose(controls, a_gpu, a_gpu);

            } t.elapsed();
            if (DEBUG_ENABLE) Log() << "transpose in " << t.last_elapsed();

            compare_matrices(controls, a_gpu, a_dcsr_cpu_tr);
        }
    }
}