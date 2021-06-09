#pragma once
#include <gtest/gtest.h>
#include <random>
#include "../dcsr/dcsr.hpp"
#include <matrices_conversions.hpp>
#include "../coo/coo_utils.hpp"
#include "cl_includes.hpp"
#include "env.hpp"
#include "memory"


struct Wrapper {
    static std::shared_ptr<clbool::Controls> controls;
    static void initControls();
};

// operations
bool test_transpose(clbool::Controls &controls, uint32_t size, uint32_t k);

bool test_submatrix(clbool::Controls &controls, uint32_t size, uint32_t k);

bool test_reduce(clbool::Controls &controls, uint32_t size, uint32_t k);

bool test_multiplication_hash(clbool::Controls &controls, uint32_t size, uint32_t k);

bool test_multiplication_merge(clbool::Controls &controls, uint32_t size, uint32_t k);

bool test_addition_coo(clbool::Controls &controls, uint32_t size_a,
                       uint32_t k_a, uint32_t k_b);

bool test_addition_csr(clbool::Controls &controls, uint32_t size, uint32_t k_a, uint32_t k_b);

bool test_kronecker_coo(clbool::Controls &controls,
                        uint32_t size_a, uint32_t size_b, uint32_t nnz_a, uint32_t nnz_b, uint32_t k);
bool test_kronecker_dcsr(clbool::Controls &controls,
                         uint32_t size_a, uint32_t size_b, uint32_t nnz_a, uint32_t nnz_b, uint32_t k);

// procedures
bool test_bitonic_sort(clbool::Controls &controls, uint32_t size);

bool test_pref_sum(clbool::Controls &controls, uint32_t size);

void test_dcsr_csr(clbool::Controls &controls, uint32_t size, uint32_t k);

void test_csr_dcsr(clbool::Controls &controls, uint32_t size, uint32_t k);
