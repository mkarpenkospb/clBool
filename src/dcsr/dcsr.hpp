#pragma once

#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_dcsr.hpp"
#include "../common/matrices_conversions.hpp"


void submatrix(Controls &controls,
               matrix_dcsr &matrix_out,
               const matrix_dcsr &matrix_in,
               uint32_t i, uint32_t j,
               uint32_t nrows, uint32_t ncols);