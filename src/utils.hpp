#pragma once

#include <cstdint>
#include "cl_defines.hpp"
#include "library_classes/controls.hpp"

// https://stackoverflow.com/a/466242
unsigned int ceil_to_power2(uint32_t v);

// https://stackoverflow.com/a/2681094
uint32_t round_to_power2 (uint32_t x);

uint32_t calculate_global_size(uint32_t work_group_size, uint32_t n);

Controls create_controls();