#include "src/clBool_tests.hpp"
#include "../../src/coo/coo.hpp"
#include <gtest/gtest.h>


TEST(clBool_operations, transpose) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 10; k < 30; ++k) {
        for (int size = 20; size < 400; size += 200) {
            ASSERT_TRUE(test_transpose(controls, size, k));
        }
    }
}

TEST(clBool_operations, submatrix) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 1000; size < 30000; size += 200) {
            for (int iter = 0; iter < 25; ++iter) {
                ASSERT_TRUE(test_submatrix(controls, size, k, iter));
            }
        }
    }
}

TEST(clBool_operations, reduce) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 1000; size < 30000; size += 200) {
            ASSERT_TRUE(test_reduce(controls, size, k));
        }
    }
}

TEST(clBool_operations, multiplication_hash) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 1000; size < 30000; size += 200) {
            ASSERT_TRUE(test_multiplication_hash(controls, size, k));
        }
    }
}

TEST(clBool_operations, multiplication_merge) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 1000; size < 30000; size += 200) {
            ASSERT_TRUE(test_multiplication_merge(controls, size, k));
        }
    }
}


TEST(clBool_operations, addition_coo) {
    clbool::Controls controls = clbool::create_controls();
    for (int size_a = 100; size_a < 10000; ++size_a) {
        for (int size_b = 100; size_b < 1000; ++size_b) {
            for (uint32_t k_a = 40; k_a < 60; ++k_a) {
                for (uint32_t k_b = 40; k_b < 60; ++k_b) {
                    ASSERT_TRUE(test_addition_coo(controls, size_a, size_b, k_a, k_b));
                }
            }
        }
    }
}

TEST(clBool_operations, kronecker_coo) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 1; k < 10; ++k) {
        uint32_t base_nnz = 1000 * k;
        ASSERT_TRUE(test_kronecker_coo(controls, 10000, 10000, base_nnz, base_nnz + 5, k));
    }
}

TEST(clBool_operations, kronecker_dcsr) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 1; k < 10; ++k) {
        uint32_t base_nnz = 1000 * k;
        ASSERT_TRUE(test_kronecker_dcsr(controls, 10000, 10000, base_nnz, base_nnz + 5, k));
    }
}

TEST(clBool_operations, example) {

    clbool::Controls controls = clbool::create_controls();
    uint32_t a_nrows = 5, a_ncols = 5, a_nnz = 6;
    std::vector<uint32_t> a_rows = {0, 0, 0, 2, 2, 4};
    std::vector<uint32_t> a_cols = {0, 1, 4, 2, 3, 2};


    uint32_t b_nrows = 5, b_ncols = 5, b_nnz = 7;
    std::vector<uint32_t> b_rows = {1, 1, 2, 3, 3, 3, 5};
    std::vector<uint32_t> b_cols = {0, 4, 2, 2, 3, 4, 2};

    clbool::matrix_coo a_coo(controls, a_nrows, a_ncols, a_nnz, a_rows.data(), a_cols.data());
    clbool::matrix_coo b_coo(controls, b_nrows, b_ncols, b_nnz, b_rows.data(), b_cols.data());

    clbool::matrix_coo c_coo;
    clbool::coo::matrix_addition(controls, c_coo, a_coo, b_coo);
}