#include "coo_tests.hpp"


#include "coo_tests.hpp"
#include "../../cl_includes.hpp"
#include "../../library_classes/controls.hpp"
#include "../../library_classes/matrix_dcsr.hpp"
#include "../../utils.hpp"
#include "../coo_utils.hpp"
#include "../dscr_matrix_multiplication.hpp"

using namespace coo_utils;
const uint32_t BINS_NUM = 38;


using namespace coo_utils;
using namespace utils;

void checkCopying() {
    Controls controls = utils::create_controls();

    uint32_t nnz_limit = 25;
    uint32_t max_size = 10;

    matrix_coo_cpu matrix_a_coo_cpu = generate_random_matrix_coo_cpu(nnz_limit, max_size);
    matrix_dcsr_cpu matrix_a_cpu = coo_to_dcsr_cpu(matrix_a_coo_cpu);

    std::cout << "matrix_a_coo_cpu: \n";
    print_matrix(matrix_a_coo_cpu);

    std::cout << "matrix_a_cpu: \n";
    print_matrix(matrix_a_cpu);

    matrix_coo a_coo = coo_utils::matrix_coo_from_cpu(controls, matrix_a_coo_cpu);

    cl::Buffer a_rows_pointers;

    /*
     * _rows_compressed -- rows array with no duplicates
     * probably we don't need it
     */
    cl::Buffer a_rows_compressed;
    uint32_t a_nzr;

    create_rows_pointers(controls, a_rows_pointers, a_rows_compressed, a_coo.rows_indices_gpu(), a_coo.nnz(), a_nzr);


    matrix_dcsr a_dcsr = matrix_dcsr(a_rows_pointers, a_rows_compressed, a_coo.cols_indices_gpu(),
                                     a_coo.nRows(), a_coo.nCols(), a_coo.nnz(), a_nzr);

    uint32_t value = 239;
    controls.queue.enqueueWriteBuffer(a_coo.cols_indices_gpu(), CL_TRUE, 0, sizeof(uint32_t) * 1, &value);
    print_gpu_buffer(controls, a_dcsr.cols_indices_gpu(), 3);

}

void simpleTestCopy() {
    Controls controls = create_controls();

    cpu_buffer a_cpu(5, 1);

    cl::Buffer a_gpu(controls.context, CL_TRUE, sizeof(uint32_t) * a_cpu.size());
    cl::Buffer a_gpu_copy(controls.context, CL_TRUE, sizeof(uint32_t) * a_cpu.size());

    controls.queue.enqueueWriteBuffer(a_gpu, CL_TRUE, 0, sizeof(uint32_t) * a_cpu.size(), a_cpu.data());
    print_gpu_buffer(controls, a_gpu_copy, a_cpu.size());
    controls.queue.enqueueCopyBuffer(a_gpu, a_gpu_copy, 0, 0, sizeof(uint32_t) * a_cpu.size());

    print_gpu_buffer(controls, a_gpu, a_cpu.size());
    print_gpu_buffer(controls, a_gpu_copy, a_cpu.size());
}

