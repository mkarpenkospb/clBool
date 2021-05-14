#include "src/clBool_tests.hpp"
#include <gtest/gtest.h>


TEST(clBool_procedures, bitonic_sort) {
    clbool::Controls controls = clbool::create_controls();
    for (uint32_t size = 10; size < 500; size += 17) {
        ASSERT_TRUE(test_bitonic_sort(controls, size));
    }
}

TEST(clBool_procedures, pref_sum) {
    clbool::Controls controls = clbool::create_controls();
    for (int size = 800000; size < 1200000; size += 10000) {
        ASSERT_TRUE(test_pref_sum(controls, size));
    }
}

TEST(clBool_procedures, devices) {
    clbool::show_devices();
}
