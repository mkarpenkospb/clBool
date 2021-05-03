#include "src/clBool_tests.hpp"
#include <gtest/gtest.h>


TEST(clBool_procedures, bitonic_sort) {
    clbool::Controls controls = clbool::utils::create_controls();
    for (uint32_t size = 0; size < 1020000; size += 2000) {
        ASSERT_TRUE(test_bitonic_sort(controls, size));
    }
}

TEST(clBool_procedures, pref_sum) {
    clbool::Controls controls = clbool::utils::create_controls();
    for (int size = 10; size < 400000; size += 100) {
        ASSERT_TRUE(test_pref_sum(controls, size));
    }
}
