#include "csr.hpp"

#include "matrix_csr.hpp"
#include "utils.hpp"
#include "kernel.hpp"
#include <cl_operations.hpp>

namespace clbool::csr {

    void matrix_addition(Controls &controls, matrix_csr &c, const matrix_csr &a, const matrix_csr &b) {

        if (a.nrows() != b.nrows() || a.ncols() != b.ncols()) {
            std::stringstream s;
            s << "Invalid matrixes size! a: " << a.nrows() << " x " << a.ncols() <<
              ", b: " << b.nrows() << " x " << b.ncols();
            CLB_RAISE(s.str(), CLBOOL_INVALID_ARGUMENT);
        }

        if (a.empty() && b.empty()) {
            c = matrix_csr(a.nrows(), a.ncols());
            return;
        }

        if (a.empty() || b.empty()) {
            const matrix_csr &empty = a.empty() ? a : b;
            const matrix_csr &filled = a.empty() ? b : a;

            if (&c == &filled) return;

            cl::Buffer rpt;
            cl::Buffer cols;
            CLB_CREATE_BUF(rpt = utils::create_buffer(controls, filled.nrows() + 1));
            CLB_CREATE_BUF(cols = utils::create_buffer(controls, filled.nnz()));

            CLB_COPY_BUF(controls.queue.enqueueCopyBuffer(filled.rpt_gpu(), rpt, 0, 0, sizeof(uint32_t) * (filled.nrows() + 1)));
            CLB_COPY_BUF(controls.queue.enqueueCopyBuffer(filled.cols_gpu(), cols, 0, 0, sizeof(uint32_t) * filled.nnz()));
            c = matrix_csr(rpt, cols, filled.nrows(), filled.ncols(), filled.nnz());
            return;
        }


        cl::Buffer c_rpt;
        CLB_CREATE_BUF(c_rpt = utils::create_buffer(controls, a.nrows() + 1));

        {
            auto add_symbolic = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                    ("csr_addition", "addition_symbolic");
            add_symbolic.set_block_size(controls.block_size);
            add_symbolic.set_work_size(controls.block_size * a.nrows());

            CLB_RUN(TIME_RUN("add_symbolic run in:",
                    add_symbolic.run(controls, a.rpt_gpu(), a.cols_gpu(), b.rpt_gpu(), b.cols_gpu(),c_rpt, a.nrows())));
        }

        uint32_t c_nnz;

        {
            START_TIMING
            prefix_sum(controls, c_rpt, c_nnz, a.nrows() + 1);
            END_TIMING("prefix_sum run in: ")
        }


        cl::Buffer c_cols;
        CLB_CREATE_BUF(c_cols = utils::create_buffer(controls, c_nnz));

        {
            auto add_numeric = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                    ("csr_addition", "addition_numeric");
            add_numeric.set_block_size(controls.block_size);
            add_numeric.set_work_size(controls.block_size * a.nrows());

            CLB_RUN(TIME_RUN("add_symbolic run in:",
                    add_numeric.run(controls, a.rpt_gpu(), a.cols_gpu(), b.rpt_gpu(), b.cols_gpu(), c_rpt, c_cols)));
        }

        c = matrix_csr(c_rpt, c_cols, a.nrows(), a.ncols(), c_nnz);
    }
}

