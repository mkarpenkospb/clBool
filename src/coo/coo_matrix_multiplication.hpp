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
                    cl::Buffer &workload_out,
                    cl::Buffer &a_rows_pointers,
                    const cl::Buffer &a_cols,
                    cl::Buffer &b_rows_compressed,
                    cl::Buffer &b_rows_pointers,
                    const cl::Buffer &b_cols,
                    uint32_t a_nzr,
                    uint32_t b_nzr);

void build_groups_and_allocate_new_matrix(Controls& controls,
                                          cl::Buffer& pre_rows_pointers,
                                          cl::Buffer& pre_cols_indices_gpu,
                                          uint32_t &pre_nnz,
                                          std::vector<cpu_buffer>& workload_groups,
                                          const cpu_buffer& cpu_workload,
                                          uint32_t a_nzr
);

uint32_t get_group(uint32_t size);

auto get_heap_kernel(Controls &controls,
                     uint32_t group_length,
                     unsigned int nnz_estimation
);

auto get_copy_one_value_kernel(Controls &controls,
                               uint32_t group_length
);


void run_kernels(Controls &controls,
                 const std::vector<cpu_buffer> &cpu_workload_groups,
                 const cpu_buffer &groups_length,
                 const cpu_buffer &groups_pointers,

                 const cl::Buffer &gpu_workload_groups,
                 cl::Buffer &nnz_estimation,

                 const cl::Buffer &pre_rows_pointers,
                 cl::Buffer &pre_cols_indices_gpu,

                 const cl::Buffer &a_rows_pointers,
                 const cl::Buffer &a_cols,

                 const cl::Buffer &b_rows_pointers,
                 const cl::Buffer &b_rows_compressed,
                 const cl::Buffer &b_cols,

                 uint32_t b_nzr

);

void write_bins_info(Controls &controls,
                     cl::Buffer &gpu_workload_groups,
                     const std::vector<cpu_buffer> &cpu_workload_groups,
                     cpu_buffer &groups_pointers,
                     cpu_buffer &groups_length
);