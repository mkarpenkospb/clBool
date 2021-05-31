#include <gtest/gtest.h>
#include "src/clBool_tests.hpp"

/*
 * Platform and device ids which throws depends on available number of platforms and devices!
 */
TEST(clBool_errors, invalid_ids) {
    EXPECT_THROW(clbool::Controls controls = clbool::create_controls(0, 1), clbool::Exception);
    EXPECT_THROW(clbool::Controls controls = clbool::create_controls(10, 0), clbool::Exception);
}

TEST(clBool_errors, invalid_kernel) {
    using namespace clbool;
    clbool::Controls controls = clbool::create_controls();
    auto prepare_positions = clbool::kernel<cl::Buffer, cl::Buffer, uint32_t>
            ("prepare_positions", "some name")
            .set_work_size(25);
    cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * 25);
    cl::Buffer array(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * 25);
    cl::Buffer size(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * 25);

    EXPECT_THROW(CHECK_RUN(prepare_positions.run(controls, positions, array, 25), 465348324), Exception);
}

TEST(clBool_errors, invalid_buffer_size) {
    using namespace clbool;
    clbool::Controls controls = clbool::create_controls();
    EXPECT_THROW(
            CHECK_CL(cl::Buffer size(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * 0),
            CLBOOL_CREATE_BUFFER_ERROR, 88211), Exception);
}