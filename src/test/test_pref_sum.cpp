#include <algorithm>
#include "coo_tests.hpp"

#include <coo_utils.hpp>
#include <cpu_matrices.hpp>
#include <cl_operations.hpp>

using namespace coo_utils;
using namespace utils;

void test_pref_sum() {
    Controls controls = create_controls();
    for (int iter = 0; iter < 10; iter ++) {
        if (DEBUG_ENABLE) *logger << "\n------------------- ITER " << iter << " -------------------------\n";
        std::cout << "\n------------------- ITER " << iter << " -------------------------\n";
        int size = 10456273;
        cpu_buffer vec(size, 0);
        utils::fill_random_buffer(vec, 345232, 4);
        if (DEBUG_ENABLE) *logger << "\n data generated for n = "<< size << "\n";

        cl::Buffer vec_gpu(controls.queue, vec.begin(), vec.end(), false);
        int prev = vec[0];
        int tmp;
        vec[0] = 0;
        timer t;
        t.restart();
        for (int i = 1; i < vec.size(); ++i) {
            tmp = vec[i];
            vec[i] = prev;
            prev += tmp;
        }
        double time = t.elapsed();
        if (DEBUG_ENABLE) *logger << "CPU scan finished in " << time << "\n";
        uint32_t total;
        t.restart();
        prefix_sum(controls, vec_gpu, total, size);
        time = t.elapsed();
        if (DEBUG_ENABLE) *logger << "DEVICE scan finished in " << time << "\n";
//        if (total != prev) {
//            throw std::runtime_error("sums are different!");
//        }

        compare_buffers(controls, vec_gpu, vec, size);
    }
}