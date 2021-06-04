#include <gtest/gtest.h>
#include "src/clBool_tests.hpp"


TEST(clBool_conversions, dcsr_csr_small) {
    clbool::Controls controls = clbool::create_controls();

    for (int size = 0; size < 4000; size += 200) {
        for (uint32_t k = 0; k < 30; ++k) {
            test_dcsr_csr(controls, size, k);
        }
    }
}

TEST(clBool_conversions, dcsr_csr_large) {
    clbool::Controls controls = clbool::create_controls();

    for (int size = 4000; size < 30000; size += 700) {
        for (uint32_t k = 30; k < 70; k += 10) {
            test_dcsr_csr(controls, size, k);
        }
    }

}

TEST(clBool_conversions, csr_dcsr_small) {
    clbool::Controls controls = clbool::create_controls();

    for (int size = 0; size < 4000; size += 200) {
        for (uint32_t k = 0; k < 30; ++k) {
            test_csr_dcsr(controls, size, k);
        }
    }
}

TEST(clBool_conversions, csr_dcsr_large) {
    clbool::Controls controls = clbool::create_controls();

    for (int size = 4000; size < 30000; size += 700) {
        for (uint32_t k = 30; k < 70; k += 10) {
            test_csr_dcsr(controls, size, k);
        }
    }

}