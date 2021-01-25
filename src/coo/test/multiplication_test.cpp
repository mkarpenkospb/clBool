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

const uint32_t BINS_NUM = 38;

void multiplicationTest() {
    Controls controls = utils::create_controls();

    uint32_t nnz_limit = 25;
    uint32_t max_size = 10;
    // TODO: сделать их одинаковыми в следующем тесте
    matrix_dcsr_cpu matrix_a_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_limit, max_size));
    matrix_dcsr_cpu matrix_b_cpu = coo_to_dcsr_cpu(generate_random_matrix_coo_cpu(nnz_limit + 1, max_size + 1));
    std::cout << "matrix_a_cpu: \n";
    print_matrix(matrix_a_cpu);
    std::cout << "matrix_b_cpu: \n";
    print_matrix(matrix_b_cpu);
    matrix_dcsr_cpu matrix_c_cpu;

    matrix_multiplication_cpu(matrix_c_cpu, matrix_a_cpu, matrix_b_cpu);





    print_matrix(matrix_c_cpu);

}

