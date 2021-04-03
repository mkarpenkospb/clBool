#include <algorithm>
#include "coo_tests.hpp"
#include <controls.hpp>
#include <utils.hpp>
#include <coo_utils.hpp>
#include <program.hpp>
using namespace utils;

void test_merge() {
    timer t;
    Controls controls = create_controls();
    for (int iter = 0; iter < 10; iter ++) {
        if (DEBUG_ENABLE)
            Log() << "\n----------------------------- ITER " << iter << " --------------------------------\n";
        std::cout << "\n----------------------------- ITER " << iter << " --------------------------------\n";
        int i = 1234234;
        int j = 3746761;

        matrix_coo_cpu_pairs matrix_res_cpu;
        matrix_coo_cpu_pairs matrix_a_cpu = coo_utils::generate_random_matrix_coo_cpu(i, 10756);
        matrix_coo_cpu_pairs matrix_b_cpu = coo_utils::generate_random_matrix_coo_cpu(j, 23341);

        if (DEBUG_ENABLE) Log() << "data generated for a_nnz ~ " << matrix_a_cpu.size()
                                  << " and b_nnz ~ " << matrix_b_cpu.size() << " \n";

        matrix_coo matrix_res_gpu;


        t.restart();
        std::merge(matrix_a_cpu.begin(), matrix_a_cpu.end(), matrix_b_cpu.begin(), matrix_b_cpu.end(),
                   std::back_inserter(matrix_res_cpu));

        double time = t.elapsed();
        if (DEBUG_ENABLE) Log() << "merge on CPU finished in " << time << " \n";

        t.restart();
        matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu);
        matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu);
        time = t.elapsed();
        if (DEBUG_ENABLE) Log() << "matrices transferred to DEVICE in " << time << " \n";

        t.restart();
        uint32_t merged_size = matrix_res_cpu.size();

        auto coo_merge = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t, uint32_t>
                ("merge_path")
                .set_needed_work_size(merged_size)
                .set_kernel_name("merge");


        cl::Buffer merged_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);
        cl::Buffer merged_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

        coo_merge.run(controls,
                      merged_rows, merged_cols,
                      matrix_a_gpu.rows_indices_gpu(), matrix_a_gpu.cols_indices_gpu(),
                      matrix_b_gpu.rows_indices_gpu(), matrix_b_gpu.cols_indices_gpu(),
                      matrix_a_gpu.nnz(), matrix_b_gpu.nnz()).wait();

        time = t.elapsed();
        if (DEBUG_ENABLE) Log() << "merge on DEVICE finished in " << time << " \n";

        cpu_buffer rows_cpu;
        cpu_buffer cols_cpu;

        coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, matrix_res_cpu);

        utils::compare_buffers(controls, merged_rows, rows_cpu, rows_cpu.size());
        utils::compare_buffers(controls, merged_cols, cols_cpu, cols_cpu.size());

    }
}