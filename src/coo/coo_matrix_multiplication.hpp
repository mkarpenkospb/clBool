#pragma once


#include "../library_classes/controls.hpp"

void prepare_positions(Controls &controls,
                       cl::Buffer &positions,
                       const cl::Buffer &rows,
                       uint32_t size
);


void set_positions(Controls &controls,
                   cl::Buffer &rows_pointers,
                   const cl::Buffer &rows,
                   cl::Buffer &positions,
                   uint32_t size);


void create_rows_pointers(Controls &controls,
                          cl::Buffer &rows_pointers_out,
                          const cl::Buffer &rows,
                          uint32_t size,
                          uint32_t new_size);

void count_workload(Controls &controls,
                    cl::Buffer workload_out,
                    cl::Buffer &a_rows_pointers,
                    const cl::Buffer &a_cols,
                    cl::Buffer &b_rows_pointers,
                    const cl::Buffer &b_cols,
                    uint32_t a_nzr,
                    uint32_t b_nzr);