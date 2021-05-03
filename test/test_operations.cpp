#include "src/clBool_tests.hpp"
#include <gtest/gtest.h>


TEST(clBool_operations, transpose) {
    clbool::Controls controls = clbool::utils::create_controls();
    for (uint32_t k = 10; k < 30; ++k) {
        for (int size = 20; size < 400; size += 200) {
            ASSERT_TRUE(test_transpose(controls, size, k));
        }
    }
}

TEST(clBool_operations, submatrix) {
    clbool::Controls controls = clbool::utils::create_controls();
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 1000; size < 30000; size += 200) {
            for (int iter = 0; iter < 25; ++iter) {
                ASSERT_TRUE(test_submatrix(controls, size, k, iter));
            }
        }
    }
}

TEST(clBool_operations, reduce) {
    clbool::Controls controls = clbool::utils::create_controls();
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 1000; size < 30000; size += 200) {
            ASSERT_TRUE(test_reduce(controls, size, k));
        }
    }
}

TEST(clBool_operations, multiplication_hash) {
    clbool::Controls controls = clbool::utils::create_controls();
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 1000; size < 30000; size += 200) {
            ASSERT_TRUE(test_multiplication_hash(controls, size, k));
        }
    }
}

TEST(clBool_operations, multiplication_merge) {
    clbool::Controls controls = clbool::utils::create_controls();
    for (uint32_t k = 40; k < 60; ++k) {
        for (int size = 1000; size < 30000; size += 200) {
            ASSERT_TRUE(test_multiplication_merge(controls, size, k));
        }
    }
}


TEST(clBool_operations, addition_coo) {
    clbool::Controls controls = clbool::utils::create_controls();
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
