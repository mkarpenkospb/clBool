#include "clBool_tests.hpp"

using namespace clbool;
using namespace clbool::coo_utils;
using namespace clbool::utils;

bool test_pref_sum(Controls &controls, uint32_t size) {
    SET_TIMER

    LOG << "------------------" << " size = " << size << " --------------------";

    utils::cpu_buffer vec(size, 0);
    utils::fill_random_buffer(vec, 3);
    vec.push_back(0);

    cl::Buffer vec_gpu(controls.queue, vec.begin(), vec.end(), false);

    int prev;

    {
        START_TIMING

        prev = vec[0];
        int tmp;
        vec[0] = 0;
        for (int i = 1; i < vec.size(); ++i) {
            tmp = vec[i];
            vec[i] = prev;
            prev += tmp;
        }

        END_TIMING("CPU prefix sum: ")
    }

    uint32_t total;

    {
        START_TIMING
        prefix_sum(controls, vec_gpu, total, size + 1);
        END_TIMING("DEVICE prefix sum: ")
    }

    if (total != prev) {
        throw std::runtime_error("sums are different!");
    }

    return compare_buffers(controls, vec_gpu, vec, size + 1);
}

bool test_bitonic_sort(Controls &controls, uint32_t size) {
    SET_TIMER

    // TODO: not true of course
    if (size == 0) return true;

    LOG << "----------------------- size = " << size << " ------------------------";

    cpu_buffer rows_cpu(size);
    cpu_buffer cols_cpu(size);

    coo_utils::fill_random_matrix(rows_cpu, cols_cpu);

    cl::Buffer rows_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
    cl::Buffer cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

    controls.queue.enqueueWriteBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_cpu.data());
    controls.queue.enqueueWriteBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_cpu.data());

    matrix_coo_cpu_pairs m_cpu;
    coo_utils::form_cpu_matrix(m_cpu, rows_cpu, cols_cpu);

    {
        START_TIMING
        std::sort(m_cpu.begin(), m_cpu.end());
        END_TIMING("sort on CPU: ")
    }
    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, m_cpu);


    {
        START_TIMING
        coo::sort_arrays(controls, rows_gpu, cols_gpu, size);
        END_TIMING("sort on DEVICE: ")
    }

    return compare_buffers(controls, rows_gpu, rows_cpu, size, "rows_gpu") &&
           compare_buffers(controls, cols_gpu, cols_cpu, size, "cols_gpu");
}


