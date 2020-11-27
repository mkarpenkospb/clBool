#pragma once

#include "../cl_defines.hpp"

void addition(
        Controls &controls,
        matrix_coo &matrix_out,
        const matrix_coo &a,
        const matrix_coo &b
);


void check_merge_correctness(
        Controls &controls,
        cl::Buffer &rows,
        cl::Buffer &cols,
        size_t merged_size
);

void merge(
        Controls &controls,
        cl::Buffer &merged_rows,
        cl::Buffer &merged_cols,
        const matrix_coo &a,
        const matrix_coo &
);

void prepare_positions(
        Controls &controls,
        cl::Buffer &positions,
        cl::Buffer &merged_rows,
        cl::Buffer &merged_cols,
        uint32_t merged_size
);

void prefix_sum(
        Controls &controls,
        cl::Buffer &positions,
        uint32_t &new_size,
        uint32_t merged_size
);


void set_positions(
        Controls &controls,
        cl::Buffer &new_rows,
        cl::Buffer &new_cols,
        cl::Buffer &merged_rows,
        cl::Buffer &merged_cols,
        cl::Buffer &positions,
        uint32_t merged_size
);

void reduce_duplicates(
        Controls &controls,
        cl::Buffer &merged_rows,
        cl::Buffer &merged_cols,
        uint32_t &new_size,
        uint32_t merged_size
);