#include <algorithm>
#include "../../cl_includes.hpp"
#include "coo_tests.hpp"
#include "../../library_classes/controls.hpp"
#include "../coo_utils.hpp"

using coo_utils::matrix_coo_cpu;
void testBitonicSort() {
    Controls controls = utils::create_controls();

    uint32_t size = 15;
    std::vector<uint32_t> rows_cpu(size);
    std::vector<uint32_t> cols_cpu(size);

    std::vector<uint32_t> rows_from_gpu(size);
    std::vector<uint32_t> cols_from_gpu(size);

    coo_utils::fill_random_matrix(rows_cpu, cols_cpu);

    matrix_coo_cpu m_cpu;

    coo_utils::form_cpu_matrix(m_cpu, rows_cpu, cols_cpu);

    std::sort(m_cpu.begin(), m_cpu.end());

    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, m_cpu);

    cl::Buffer rows_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
    cl::Buffer cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

    controls.queue.enqueueWriteBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_cpu.data());
    controls.queue.enqueueWriteBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_cpu.data());

    sort_arrays(controls, rows_gpu, cols_gpu, size);

    controls.queue.enqueueReadBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_from_gpu.data());
    controls.queue.enqueueReadBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_from_gpu.data());

    if (rows_from_gpu == rows_cpu && cols_from_gpu == cols_cpu) {
        std::cout << "correct sort" << std::endl;
    } else {
        std::cerr << "incorrect sort" << std::endl;
    }
}

