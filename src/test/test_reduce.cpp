#include "coo_tests.hpp"

#include <random>
#include "../dcsr/dcsr.hpp"
#include <matrices_conversions.hpp>
#include "../coo/coo_utils.hpp"

using namespace coo_utils;
using namespace utils;

void test_reduce() {
    Controls controls = create_controls();
    timer t;
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 3000; size < 10000; size += 200) {
            uint32_t max_size = size;
            uint32_t nnz_max = std::max(10u, max_size * k);

            if (DEBUG_ENABLE)
                Log() << " ------------------------------- k = " << k << ", size = " << size
                      << " -------------------------------------------\n"
                      << "max_size = " << size << ", nnz_max = " << nnz_max;

            matrix_dcsr_cpu a_cpu = coo_to_dcsr_cpu(generate_coo_cpu(nnz_max, max_size));

            matrix_dcsr a_gpu;
            t.restart(); {

                a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);

            } t.elapsed();
            if (DEBUG_ENABLE) Log() << "matrix_dcsr_from_cpu in " << t.last_elapsed();


            t.restart(); {

                reduce(a_cpu, a_cpu);

            } t.elapsed();
            if (DEBUG_ENABLE) Log() << "reduce on CPU finished in " << t.last_elapsed();


            t.restart(); {

                reduce(controls, a_gpu, a_gpu);

            } t.elapsed();
            if (DEBUG_ENABLE) Log() << "reduce on DEVICE finished in " << t.last_elapsed();

            compare_matrices(controls, a_gpu, a_cpu);
        }
    }
}