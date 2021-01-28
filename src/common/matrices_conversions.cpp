#include "matrices_conversions.hpp"

matrix_coo dcsr_to_coo(Controls &controls, matrix_dcsr &a) {
    cl::Buffer c_rows_indices(controls.context, CL_MEM_READ_WRITE, sizeof(matrix_dcsr::index_type) * a.nzr());

    auto dscr_to_coo_kernel = program<cl::Buffer, cl::Buffer, cl::Buffer>("../src/cl/dscr_to_coo.cl")
            .set_kernel_name("dscr_to_coo")
            .set_block_size(64)
            .set_needed_work_size(a.nzr() * 64);

    dscr_to_coo_kernel.run(controls, a.rows_pointers_gpu(), a.rows_compressed_gpu(), c_rows_indices);
    return matrix_coo(a.nRows(), a.nCols(), a.nnz(), c_rows_indices, a.cols_indices_gpu());
}

namespace {
    void create_rows_pointers(Controls &controls,
                              cl::Buffer &rows_pointers_out,
                              cl::Buffer &rows_compressed_out,
                              const cl::Buffer &rows,
                              uint32_t size,
                              uint32_t &nzr // non zero rows
    ) {

        cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

        auto prepare_positions = program<cl::Buffer, cl::Buffer, uint32_t>("../src/cl/prepare_positions.cl")
                .set_kernel_name("prepare_array_for_rows_positions")
                .set_needed_work_size(size);
        prepare_positions.run(controls, positions, rows, size);

        prefix_sum(controls, positions, nzr, size);

        cl::Buffer rows_pointers(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (nzr + 1));
        cl::Buffer rows_compressed(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * nzr);

        auto set_positions = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>(
                "../src/cl/set_positions.cl")
                .set_kernel_name("set_positions_rows")
                .set_needed_work_size(size);

        set_positions.run(controls, rows_pointers, rows_compressed, rows, positions, size, nzr);

        rows_pointers_out = std::move(rows_pointers);
        rows_compressed_out = std::move(rows_compressed);
    }
}

matrix_dcsr coo_to_dcsr_gpu(Controls &controls, const matrix_coo &a) {
    cl::Buffer rows_pointers;
    cl::Buffer rows_compressed;
    uint32_t nzr;
    create_rows_pointers(controls, rows_pointers, rows_compressed, a.rows_indices_gpu(), a.nnz(), nzr);

    return matrix_dcsr(rows_pointers, rows_compressed, a.rows_indices_gpu(),
                       a.nRows(), a.nCols(), a.nnz(), nzr
    );
}