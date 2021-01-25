#include "coo_tests.hpp"


#include "coo_tests.hpp"
#include "../../cl_includes.hpp"
#include "../../library_classes/controls.hpp"
#include "../../utils.hpp"
#include "../coo_utils.hpp"
#include "../dscr_matrix_multiplication.hpp"

using namespace coo_utils;

void testCountWorkload() {
    Controls controls = utils::create_controls();

    uint32_t nnz_limit = 1'000'000;
    uint32_t max_size = 1'024;
    matrix_dcsr_cpu a_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_limit, max_size));
    matrix_dcsr_cpu b_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_limit + 1, max_size));

    if (nnz_limit < 50) {
        coo_utils::print_matrix(a_cpu);
        coo_utils::print_matrix(b_cpu);
    }

    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
    matrix_dcsr b_gpu = matrix_dcsr_from_cpu(controls, b_cpu, max_size);

    // get workload from gpu

    cl::Buffer nnz_estimation;
    count_workload(controls, nnz_estimation, a_gpu, b_gpu);

    std::cout << "finish gpu counting" << std::endl;

    // get workload from cpu
    cpu_buffer nnz_estimation_cpu(a_gpu.nzr());

    coo_utils::get_workload(nnz_estimation_cpu, a_cpu, b_cpu);

    std::cout << "finish cpu counting" << std::endl;

    // compare buffers
    utils::compare_buffers(controls, nnz_estimation, nnz_estimation_cpu, a_gpu.nzr());
}

