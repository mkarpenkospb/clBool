#pragma once


#include "../library_classes/controls.hpp"

typedef std::vector<uint32_t> cpu_buffer;


void prepare_positions(Controls &controls,
                       cl::Buffer &positions,
                       const cl::Buffer &rows,
                       uint32_t size
);


void set_positions(Controls &controls,
                   cl::Buffer &rows_pointers,
                   cl::Buffer &rows_compressed,
                   const cl::Buffer &rows,
                   const cl::Buffer &positions,
                   uint32_t size,
                   uint32_t nzr
);


void create_rows_pointers(Controls &controls,
                          cl::Buffer &rows_pointers_out,
                          cl::Buffer &rows_compressed_out,
                          const cl::Buffer &rows,
                          uint32_t size,
                          uint32_t &nzr);

void count_workload(Controls &controls,
                    cl::Buffer workload_out,
                    cl::Buffer &a_rows_pointers,
                    const cl::Buffer &a_cols,
                    cl::Buffer &b_rows_compressed,
                    cl::Buffer &b_rows_pointers,
                    const cl::Buffer &b_cols,
                    uint32_t a_nzr,
                    uint32_t b_nzr);

void build_groups(std::vector<cpu_buffer>& workload_groups,
                  const cpu_buffer& cpu_workload);

uint32_t get_group(uint32_t size);