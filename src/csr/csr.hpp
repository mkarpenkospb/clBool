#pragma once

#include <controls.hpp>
#include <matrix_csr.hpp>

namespace clbool::csr {
    void matrix_addition(Controls &controls, matrix_csr &c, const matrix_csr &a, const matrix_csr &b);

}