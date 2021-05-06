#include "src/clBool_tests.hpp"
#include <gtest/gtest.h>


TEST(clBool_pieces, new_merge) {
    clbool::Controls controls = clbool::create_controls();
    for (int size_a = 256; size_a < 30000; size_a += 500) {
        for (int size_b = 256; size_b < 30000; size_b += 500) {
            ASSERT_TRUE(test_new_merge(controls, size_a, size_b));
        }
    }
}

