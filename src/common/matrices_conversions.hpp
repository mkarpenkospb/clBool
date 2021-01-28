#pragma once

#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_dcsr.hpp"
#include "../library_classes/program.hpp"
#include "cl_operations.hpp"


matrix_coo dcsr_to_coo(Controls &controls, matrix_dcsr &a);
matrix_dcsr coo_to_dcsr_gpu(Controls &controls, const matrix_coo &a);