#include "coo_tests.hpp"
#include "../common/cl_includes.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../coo/coo_utils.hpp"
#include "../coo/coo_matrix_addition.hpp"


void testMatrixAddition() {
    timer t;
    Controls controls = utils::create_controls("set_positions.aocx");
    controls.FPGA_BINARIES = "src/cl/fpga/addition_kernels/";
    for (int iter = 0; iter < 20; iter++) {
        if (DEBUG_ENABLE)
            *logger << "\n----------------------------- ITER " << iter << " --------------------------------\n";

        int i = 3425433;
        int j = 234342;

        matrix_coo_cpu_pairs matrix_res_cpu;
        matrix_coo_cpu_pairs matrix_a_cpu = coo_utils::generate_random_matrix_coo_cpu(i, 34233);
        matrix_coo_cpu_pairs matrix_b_cpu = coo_utils::generate_random_matrix_coo_cpu(j, 44321);

        if (DEBUG_ENABLE) *logger << "data generated for a_nnz ~ " << matrix_a_cpu.size()
                                    << " and b_nnx ~ " << matrix_b_cpu.size();

        matrix_coo matrix_res_gpu;
        t.restart();
        matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu);
        matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu);
        double time = t.elapsed();
        if (DEBUG_ENABLE) *logger << "matrices transferred to DEVICE in " << time;

        t.restart();
        coo_utils::matrix_addition_cpu(matrix_res_cpu, matrix_a_cpu, matrix_b_cpu);
        time = t.elapsed();
        if (DEBUG_ENABLE) *logger << "matrix addition on CPU finished in " << time;

        t.restart();
        matrix_addition(controls, matrix_res_gpu, matrix_a_gpu, matrix_b_gpu);
        time = t.elapsed();
        if (DEBUG_ENABLE) *logger << "matrix addition on DEVICE finished in " << time;

        cpu_buffer rows_cpu;
        cpu_buffer cols_cpu;

        coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, matrix_res_cpu);

        // First, compare nnz

        if (matrix_res_gpu.nnz() != rows_cpu.size()) {
            throw std::runtime_error("nnz of CPU and DEVICE are different: \n   CPU: " + std::to_string(rows_cpu.size())
            + "\n   GPU: " + std::to_string(matrix_res_gpu.nnz()));
        }

        utils::compare_buffers(controls, matrix_res_gpu.rows_indices_gpu(), rows_cpu, rows_cpu.size());
        utils::compare_buffers(controls, matrix_res_gpu.cols_indices_gpu(), cols_cpu, cols_cpu.size());
    }

}
