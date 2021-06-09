#pragma once
#include <gtest/gtest.h>
#include "clBool_tests.hpp"

#define CLBOOL_GTEST_MAIN                       \
    int main(int argc, char **argv)             \
    {                                           \
        testing::InitGoogleTest(&argc, argv);   \
        return RUN_ALL_TESTS();                 \
    }
