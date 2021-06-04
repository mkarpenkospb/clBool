#include "src/clBool_tests.hpp"
#include "../../src/coo/coo.hpp"
#include "../../src/csr/csr.hpp"
#include <gtest/gtest.h>

// ------------------------ transpose -------------------------------

TEST(clBool_operations, transpose_small) {
    clbool::Controls controls = clbool::create_controls();
    for (int size = 0; size < 100; size+=5) {
        for (uint32_t k = 0; k < 20; k++) {
            ASSERT_TRUE(test_transpose(controls, size, k));
        }
    }
}

TEST(clBool_operations, transpose_medium) {
    clbool::Controls controls = clbool::create_controls();
    for (int size = 100; size < 1000; size+=50) {
        for (uint32_t k = 10; k < 30; k+=5) {
            ASSERT_TRUE(test_transpose(controls, size, k));
        }
    }
}

TEST(clBool_operations, transpose_large) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t size = 1000; size < 10000; size += 600) {
        for (uint32_t k = 30; k < 70; k += 10) {
            ASSERT_TRUE(test_transpose(controls, size, k));
        }
    }
}

// ------------------------ submatrix -------------------------------

TEST(clBool_operations, submatrix) {
    clbool::Controls controls = clbool::create_controls();
    for (int size = 0; size < 30000; size += 200) {
        for (uint32_t k = 0; k < 60; ++k) {
            ASSERT_TRUE(test_submatrix(controls, size, k));
        }
    }
}

TEST(clBool_operations, reduce) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 0; k < 60; ++k) {
        for (int size = 0; size < 30000; size += 200) {
            ASSERT_TRUE(test_reduce(controls, size, k));
        }
    }
}

TEST(clBool_operations, multiplication_hash) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 50; k < 60; k += 5) {
        for (int size = 3000; size < 4000; size += 200) {
            ASSERT_TRUE(test_multiplication_hash(controls, size, k));
        }
    }
}

TEST(clBool_operations, multiplication_merge) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 0; k < 20; ++k) {
        for (int size = 0; size < 1000; size += 200) {
            ASSERT_TRUE(test_multiplication_merge(controls, size, k));
        }
    }
}


TEST(clBool_operations, addition_coo) {
    clbool::Controls controls = clbool::create_controls();

    // SMALL

    for (int size = 0; size < 5000; size += 200) {
        for (uint32_t k_a = 0; k_a < 20; k_a += 5) {
            for (uint32_t k_b = 0; k_b < 20; k_b += 5) {
                ASSERT_TRUE(test_addition_coo(controls, size, k_a, k_b));
            }
        }
    }

    // AVG


    // LARGE
}

TEST(clBool_operations, addition_csr) {
    clbool::Controls controls = clbool::create_controls();
    for (int size = 7000; size < 10000; size += 10) {
        for (uint32_t k_a = 20; k_a < 40; k_a += 5) {
            for (uint32_t k_b = 20; k_b < 40; k_b += 5) {
                for (uint32_t i = 0; i < 10; ++i) {
                    ASSERT_TRUE(test_addition_csr(controls, size, k_a, k_b));
                }
            }
        }
    }
}

TEST(clBool_operations, kronecker_coo) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 0; k < 10; ++k) {
        uint32_t base_nnz = 1000 * k;
        ASSERT_TRUE(test_kronecker_coo(controls, 10000, 10000, base_nnz, base_nnz, k));
    }
}

TEST(clBool_operations, kronecker_dcsr) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t k = 0; k < 10; ++k) {
        uint32_t base_nnz = 1000 * k;
        ASSERT_TRUE(test_kronecker_dcsr(controls, 10000, 10000, base_nnz, base_nnz, k));
    }
}

TEST(clBool_operations, example) {

   clbool::Controls controls = clbool::create_controls();

    // ----------------------------- COO ------------------------------------
    uint32_t a_nrows = 5, a_ncols = 5, a_nnz = 6;
    std::vector<uint32_t> a_rows = {0, 0, 0, 2, 2, 4};
    std::vector<uint32_t> a_cols = {0, 1, 4, 2, 3, 2};


    uint32_t b_nrows = 5, b_ncols = 5, b_nnz = 7;
    std::vector<uint32_t> b_rows = {1, 1, 2, 3, 3, 3, 5};
    std::vector<uint32_t> b_cols = {0, 4, 2, 2, 3, 4, 2};

    clbool::matrix_coo a_coo(controls, a_rows.data(), a_cols.data(),  a_nrows, a_ncols, a_nnz);
    clbool::matrix_coo b_coo(controls, b_rows.data(), b_cols.data(), b_nrows, b_ncols, b_nnz);

    clbool::matrix_coo c_coo;
    clbool::coo::matrix_addition(controls, c_coo, a_coo, b_coo);
    clbool::coo::kronecker_product(controls, c_coo, a_coo, b_coo);


    // ----------------------------- DCSR --------------------------------

    clbool::matrix_dcsr a_dcsr = clbool::coo_to_dcsr_shallow(controls, a_coo);
    clbool::matrix_dcsr b_dcsr = clbool::coo_to_dcsr_shallow(controls, b_coo);

    clbool::matrix_dcsr c_dcsr;
    clbool::dcsr::matrix_multiplication_hash(controls, c_dcsr, a_dcsr, b_dcsr);
    clbool::dcsr::kronecker_product(controls, c_dcsr, a_dcsr, b_dcsr);
    clbool::dcsr::reduce(controls, c_dcsr, a_dcsr);
    clbool::dcsr::submatrix(controls, c_dcsr, a_dcsr, 0, 2, 3, 2);

    // ----------------------------- CSR --------------------------------

    clbool::matrix_csr a_csr = clbool::dcsr_to_csr(controls, a_dcsr);
    clbool::matrix_csr b_csr = clbool::dcsr_to_csr(controls, b_dcsr);

    clbool::matrix_csr c_csr;
    clbool::csr::matrix_addition(controls, c_csr, a_csr, b_csr);
}