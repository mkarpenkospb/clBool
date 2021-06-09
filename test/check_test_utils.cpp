#include <gtest/gtest.h>
#include "src/clBool_tests.hpp"
#include "utils.hpp"
#include "matrices_conversions.hpp"

#include "cpu_matrices.hpp"

// https://stackoverflow.com/a/33059261
bool are_equal(const clbool::cpu_buffer &a, const clbool::cpu_buffer &b) {
    bool equal = a.size() == b.size();
    EXPECT_EQ(a.size(), b.size()) << "Lengths of vectors a and b are different";
    uint32_t size = a.size();
    for (uint32_t i = 0; i < size; ++i) {
        if (a[i] != b[i]) equal = false;
        EXPECT_EQ(a[i], b[i]) << "Vectors a and b differ at index " << i;
    }
    return equal;
}


TEST(clBool_check_utils, csr_cpu_from_pairs) {
    clbool::Controls controls = clbool::create_controls();

    uint32_t m = 5, n = 7;
    clbool::matrix_coo_cpu_pairs pairs = {{0, 0}, {0, 1}, {0, 3},
                                          {2, 1}, {2, 3}, {2, 5},
                                          {3, 1}, {3, 4}, {3, 5}};
    clbool::matrix_csr_cpu expected = clbool::matrix_csr_cpu(
            {0, 3, 3, 6, 9, 9},
            {0, 1, 3,
             1, 3, 5,
             1, 4, 5},
             5, 7);

    clbool::matrix_csr_cpu result = clbool::csr_cpu_from_pairs(pairs, m, n);

    EXPECT_TRUE(are_equal(expected.rpt(), result.rpt())) << "rpt are different";
    EXPECT_TRUE(are_equal(expected.cols(), result.cols())) << "cols are different";
}