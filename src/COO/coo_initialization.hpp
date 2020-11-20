#pragma once

#include "../cl_defines.hpp"
#include "../library_classes/controls.hpp"
void check_correctness(const std::vector<uint32_t>& rows,
                       const std::vector<uint32_t>& cols);

void sort_arrays(Controls& controls, cl::Buffer &rows_gpu, cl::Buffer &cols_gpu, uint32_t n);

void fill_random_matrix(std::vector<uint32_t>& rows, std::vector<uint32_t>& cols);



