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

TEST(clBool_procedures, coo_constructor) {
    clbool::Controls controls = clbool::create_controls();

    std::vector<uint32_t> rows {0, 0, 1, 1, 3, 4, 4, 5, 5, 5, 6};
    std::vector<uint32_t> cols {1, 2, 0, 3, 5, 2, 2, 0, 3, 3, 5};
    uint32_t nrows = 7;
    uint32_t ncols = 7;
    uint32_t nnz = rows.size();

    clbool::matrix_coo mCoo(controls, rows.data(), cols.data(),  nrows, ncols, nnz, true, false);
    ASSERT_TRUE(mCoo.nnz() == nnz - 2);
}

TEST(clBool_procedures, double_initialisation) {
    auto controls = std::make_shared<clbool::Controls>(clbool::create_controls());

    std::vector<uint32_t> rows {0, 0, 1, 1, 3, 4, 4, 5, 5, 5, 6};
    std::vector<uint32_t> cols {1, 2, 0, 3, 5, 2, 2, 0, 3, 3, 5};
    uint32_t nrows = 7;
    uint32_t ncols = 7;
    uint32_t nnz = rows.size();

    clbool::matrix_coo mCoo(*controls, rows.data(), cols.data(), nrows, ncols, nnz, true, false);
    ASSERT_TRUE(mCoo.nnz() == nnz - 2);

    controls.reset();
    controls = std::make_shared<clbool::Controls>(clbool::create_controls());

    clbool::matrix_coo mCoo_2(*controls, rows.data(), cols.data(), nrows, ncols, nnz, true, false);
    ASSERT_TRUE(mCoo.nnz() == nnz - 2);

}