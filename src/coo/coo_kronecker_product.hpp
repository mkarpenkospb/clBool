#pragma once
#include <controls.hpp>
#include <matrix_coo.hpp>

void kronecker_product(Controls &controls,
                       matrix_coo &matrix_out,
                       const matrix_coo &matrix_a,
                       const matrix_coo &matrix_b
);

