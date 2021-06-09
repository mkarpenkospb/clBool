#include "src/tests.hpp"
#include "src/clBool_tests.hpp"


TEST(clBool_conversions, dcsr_csr_small) {
    Wrapper::initControls();

    for (int size = 0; size < 4000; size += 200) {
        for (uint32_t k = 0; k < 30; ++k) {
            test_dcsr_csr(*Wrapper::controls, size, k);
        }
    }
}

TEST(clBool_conversions, dcsr_csr_large) {
    Wrapper::initControls();

    for (int size = 4000; size < 30000; size += 700) {
        for (uint32_t k = 30; k < 70; k += 10) {
            test_dcsr_csr(*Wrapper::controls, size, k);
        }
    }

}

TEST(clBool_conversions, csr_dcsr_small) {
    Wrapper::initControls();

    for (int size = 0; size < 2000; size ++) {
        for (uint32_t k = 0; k < 30; ++k) {
            test_csr_dcsr(*Wrapper::controls, size, k);
        }
    }
}

TEST(clBool_conversions, csr_dcsr_large) {
    Wrapper::initControls();

    for (int size = 4000; size < 30000; size += 700) {
        for (uint32_t k = 30; k < 70; k += 10) {
            test_csr_dcsr(*Wrapper::controls, size, k);
        }
    }

}


CLBOOL_GTEST_MAIN
