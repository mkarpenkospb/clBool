#include "clBool_tests.hpp"

#include "utils.hpp"

using namespace clbool;

bool compare_cpu_buffers(const cpu_buffer &a, const cpu_buffer &b) {
    bool res = true;
    EXPECT_TRUE(res &= a.size() == b.size()) << "a.size(): " << a.size() << ", b.size(): " << b.size();
    uint32_t size = std::min(a.size(), b.size());

    for (uint32_t i = 0; i < size; ++i) {
        if (a[i] != b[i]) {
            uint32_t start = std::max(0, (int) i - 10);
            uint32_t stop = std::min(size, i + 10);
            std::cerr << "buffers are different " << std::endl
                      << "{ i: (a[i], b[i]) }" << std::endl;
            for (uint32_t j = start; j < stop; ++j) {
                if (j == i) {
                std::cerr << " !!! " << j << ": (" << a[j] << ", " << b[j] << "), ";
                } else {
                    std::cerr << j << ": (" << a[j] << ", " << b[j] << "), ";
                }
            }
            std::cerr << std::endl;
            return false;
        }
    }
    return res;
}

bool compare_matrices(const matrix_csr_cpu &a, const matrix_csr_cpu &b ) {
    bool res = true;
    EXPECT_TRUE(res &= (a.ncols() == b.ncols()));
    EXPECT_TRUE(res &= (a.nrows() == b.nrows()));
    EXPECT_TRUE(res &= compare_cpu_buffers(a.cols(), b.cols()));
    EXPECT_TRUE(res &= compare_cpu_buffers(a.rpt(), b.rpt()));
    return res;
}

bool compare_matrices(const matrix_dcsr_cpu &a, const matrix_dcsr_cpu &b ) {
    bool res = true;
    EXPECT_TRUE(res &= (a.cols() == b.cols()));
    EXPECT_TRUE(res &= compare_cpu_buffers(a.rpt(), b.rpt()));
    EXPECT_TRUE(res &= compare_cpu_buffers(a.rows(), b.rows()));
    return res;
}

void test_dcsr_csr(Controls &controls, uint32_t size, uint32_t k) {

    uint32_t max_size = size;
    uint32_t nnz_max = max_size * k;

    matrix_coo_cpu_pairs coo_pairs = coo_utils::generate_coo_pairs_cpu(nnz_max, max_size);

    std::cout << "------------------------------------" << "size: " << size << ", nnz: " << coo_pairs.size()
    << "----------------------------------------\n";

    matrix_dcsr a_dcsr = matrix_dcsr_from_cpu(controls, coo_utils::coo_pairs_to_dcsr_cpu(coo_pairs), size);
    matrix_csr a_csr = csr_from_cpu(controls, csr_cpu_from_pairs(coo_pairs, size, size));

    matrix_csr a_converted = dcsr_to_csr(controls, a_dcsr);

    EXPECT_TRUE(compare_matrices(matrix_csr_from_gpu(controls, a_converted), matrix_csr_from_gpu(controls, a_csr)));
}


void test_csr_dcsr(Controls &controls, uint32_t size, uint32_t k) {

    uint32_t max_size = size;
    uint32_t nnz_max = max_size * k;

    matrix_coo_cpu_pairs coo_pairs = coo_utils::generate_coo_pairs_cpu(nnz_max, max_size);

    std::cout << "------------------------------------" << "size: " << size << ", k: " << k << ", nnz: " << coo_pairs.size()
    << "----------------------------------------\n";

    matrix_dcsr a_dcsr = matrix_dcsr_from_cpu(controls, coo_utils::coo_pairs_to_dcsr_cpu(coo_pairs), size);
    matrix_csr a_csr = csr_from_cpu(controls, csr_cpu_from_pairs(coo_pairs, size, size));

    matrix_dcsr a_converted = csr_to_dcsr(controls, a_csr);

    EXPECT_TRUE(compare_matrices(matrix_dcsr_from_gpu(controls, a_converted), matrix_dcsr_from_gpu(controls, a_dcsr)));
}

