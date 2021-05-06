#include "dcsr.hpp"
#include "utils.hpp"
#include "cl_operations.hpp"
#include "../cl/headers/dcsr_kronecker.h"
#include <cassert>

namespace clbool::dcsr {

    void kronecker_product(Controls &controls,
                           matrix_dcsr& matrix_c,
                           const matrix_dcsr& matrix_a,
                           const matrix_dcsr& matrix_b) {

        uint32_t c_nnz = matrix_a.nnz() * matrix_b.nnz();
        uint32_t c_nzr = matrix_a.nzr() * matrix_b.nzr();
        uint32_t c_nrows = matrix_a.nrows() * matrix_b.nrows();
        uint32_t c_ncols = matrix_a.ncols() * matrix_b.ncols();

        if (c_nnz == 0) {
            matrix_c =  matrix_dcsr(c_nrows, c_ncols);
            return;
        }

        cl::Buffer c_rpt(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (c_nzr + 1));
        cl::Buffer c_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_nzr);
        cl::Buffer c_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_nnz);

        //  -------------------- form rpt and rows -------------------------------

        auto cnt_nnz = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        cl::Buffer, cl::Buffer, uint32_t, uint32_t, uint32_t>(dcsr_kronecker_kernel, dcsr_kronecker_kernel_length);
        cnt_nnz.set_needed_work_size(c_nzr);
        cnt_nnz.set_kernel_name("count_nnz_per_row");

        cnt_nnz.run(controls, c_rpt, c_rows,
                    matrix_a.rpt_gpu(), matrix_b.rpt_gpu(),
                    matrix_a.rows_gpu(), matrix_b.rows_gpu(),
                    c_nzr, matrix_b.nzr(), matrix_b.nrows());

        uint32_t total_sum;

        // c_rpt becomes an array of pointers after exclusive pref sum
        prefix_sum(controls, c_rpt, total_sum, c_nzr + 1);
        assert(total_sum == c_nnz);

        // -------------------- form cols -------------------------------

        auto kronecker = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        cl::Buffer, cl::Buffer, uint32_t, uint32_t, uint32_t , uint32_t>(dcsr_kronecker_kernel, dcsr_kronecker_kernel_length);
        kronecker.set_needed_work_size(c_nnz)
        .set_kernel_name("calculate_kronecker_product");

        kronecker.run(controls, c_rpt, c_cols, matrix_a.rpt_gpu(), matrix_b.rpt_gpu(),
                      matrix_a.cols_gpu(), matrix_b.cols_gpu(),
                      matrix_b.nzr(),
                      c_nnz, c_nzr, matrix_b.ncols()).wait();

        matrix_c = matrix_dcsr(std::move(c_rpt), std::move(c_rows), std::move(c_cols),
                               c_nrows, c_ncols, c_nnz, c_nzr);

    }
}