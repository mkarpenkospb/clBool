#include "csr.hpp"

#include "matrix_csr.hpp"
#include "utils.hpp"
#include "kernel.hpp"
#include <cl_operations.hpp>
#include <sstream>

namespace clbool::csr {
    uint32_t BIN_NUM = 4;

    uint32_t get_bin_id(uint32_t row_size) {
        if (row_size == 0) return 0;
        if (row_size <= 64) return 1;
        if (row_size <= 128) return 2;
        return 3;
    }

    uint32_t get_block_size(uint32_t bin_id) {
        if (bin_id == 0) return 128;
        if (bin_id == 1) return 64;
        if (bin_id == 2) return 128;
        if (bin_id == 3) return  256;
        std::stringstream s;
        s << "Invalid bin id " << bin_id << ", possible values: 1--3.";
        CLB_RAISE(s.str(), CLBOOL_INVALID_ARGUMENT);
    }


    void estimate_load(Controls &controls, cl::Buffer &estimation,
                       const matrix_csr &a, const matrix_csr &b) {
        CLB_CREATE_BUF(estimation = utils::create_buffer(controls, a.nrows()));
        auto estimate = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("csr_addition", "estimate_load");
        estimate.set_block_size(controls.max_wg_size);
        estimate.set_work_size(a.nrows());

        CLB_RUN(estimate.run(controls, a.rpt_gpu(), b.rpt_gpu(), estimation, a.nrows()))
    }

    void make_bins(Controls &controls, cl::Buffer &estimation, cpu_buffer &bins_offset, uint32_t nrows) {
        cpu_buffer estimation_cpu(nrows);
        CLB_READ_BUF(controls.queue.enqueueReadBuffer(estimation, true, 0, sizeof (uint32_t) * nrows,
                                                      estimation_cpu.data()));
        std::vector<cpu_buffer> permutation_cpu(BIN_NUM);
        bins_offset.resize(BIN_NUM + 1, 0);
        for (uint32_t i = 0; i < nrows; ++i) {
            uint32_t bin_id = get_bin_id(estimation_cpu[i]);
            permutation_cpu[bin_id].push_back(i);
            bins_offset[bin_id] ++;
        }
        uint32_t accum = 0;
        uint32_t tmp = 0;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < BIN_NUM; ++i) {
            if (bins_offset[i] != 0) {
                CLB_WRITE_BUF(controls.queue.enqueueWriteBuffer(estimation, true,
                                                                offset * sizeof(uint32_t), sizeof (uint32_t) * bins_offset[i],
                                                      permutation_cpu[i].data()));
                offset += bins_offset[i];
            }
            tmp = bins_offset[i];
            bins_offset[i] = accum;
            accum += tmp;
        }
        bins_offset[BIN_NUM] = accum;
    }

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

        // ---------------------------------- estimate load -----------------------------------
        cl::Buffer estimation;
        estimate_load(controls, estimation, a, b);

        cpu_buffer bins_offset;
        make_bins(controls, estimation, bins_offset, a.nrows());

        cl::Buffer c_rpt;
        CLB_CREATE_BUF(c_rpt = utils::create_buffer(controls, a.nrows() + 1));

        auto init_with_zeroes = kernel<cl::Buffer, uint32_t>
                ("csr_addition", "init_rpt");
        init_with_zeroes.set_work_size(a.nrows() + 1);
        init_with_zeroes.set_block_size(controls.max_wg_size);
        CLB_RUN(init_with_zeroes.run(controls, c_rpt, a.nrows() + 1));

        {
            std::vector<cl::Event> events;
            auto add_symbolic = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                                uint32_t, cl::Buffer, uint32_t>
                                    ("csr_addition", "addition_symbolic");
            add_symbolic.set_async(true);

            // 0 cat be ignored
            for (uint32_t i = 1; i < BIN_NUM; ++i) {
                uint32_t bin_size = bins_offset[i + 1] - bins_offset[i];
                if (bin_size == 0) continue;
                uint32_t bs = get_block_size(i);
                add_symbolic.set_block_size(bs);
                add_symbolic.set_work_size(bs * bin_size);
                cl::Event ev;
                CLB_RUN(ev = add_symbolic.run(controls, a.rpt_gpu(), a.cols_gpu(), b.rpt_gpu(), b.cols_gpu(),c_rpt, a.nrows(),
                                         estimation, bins_offset[i]));
                events.push_back(ev);
            }

            CLB_WAIT(cl::WaitForEvents(events));
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
            std::vector<cl::Event> events;
            auto add_numeric = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                                                            cl::Buffer, uint32_t>
                        ("csr_addition", "addition_numeric");
            add_numeric.set_async(true);

            for (uint32_t i = 1; i < BIN_NUM; ++i) {
                uint32_t bin_size = bins_offset[i + 1] - bins_offset[i];
                if (bin_size == 0) continue;
                uint32_t bs = get_block_size(i);
                add_numeric.set_block_size(bs);
                add_numeric.set_work_size(bs * bin_size);
                cl::Event ev;
                CLB_RUN(ev = add_numeric.run(controls, a.rpt_gpu(), a.cols_gpu(), b.rpt_gpu(), b.cols_gpu(), c_rpt, c_cols,
                                        estimation, bins_offset[i]));
                events.push_back(ev);
            }

            CLB_WAIT(cl::WaitForEvents(events));
        }

        c = matrix_csr(c_rpt, c_cols, a.nrows(), a.ncols(), c_nnz);
    }
}

