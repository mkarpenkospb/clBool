#include <cl_operations.hpp>
#include "matrix_coo.hpp"
#include "kernel.hpp"

namespace clbool {
    matrix_coo::matrix_coo(index_type nrows,
                           index_type ncols)
            : matrix_base(nrows, ncols, 0) {}

    matrix_coo::matrix_coo(index_type nrows,
                           index_type ncols,
                           index_type nnz,
                           cl::Buffer &rows,
                           cl::Buffer &cols)
            : matrix_base(nrows, ncols, nnz), _rows(rows), _cols(cols) {}


    matrix_coo::matrix_coo(Controls &controls,
                           index_type nrows,
                           index_type ncols,
                           index_type nnz,
                           const index_type *rows_indices,
                           const index_type *cols_indices,
                           bool sorted,
                           bool noDuplicates
    )
            : matrix_base(nrows, ncols, nnz) {

        if (_nnz == 0) return;

        CHECK_CL(
                _rows = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * nnz),
                CLBOOL_CREATE_BUFFER_ERROR, 279791);
        CHECK_CL(
                _cols = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(index_type) * nnz),
                CLBOOL_CREATE_BUFFER_ERROR, 93270712);

        CHECK_CL(
                controls.queue.enqueueWriteBuffer(_rows, CL_TRUE, 0, sizeof(index_type) * nnz, rows_indices),
                CLBOOL_WRITE_BUFFER_ERROR, 9879701);
        CHECK_CL(
                controls.queue.enqueueWriteBuffer(_cols, CL_TRUE, 0, sizeof(index_type) * nnz, cols_indices),
                CLBOOL_WRITE_BUFFER_ERROR, 973271);

        if (!sorted) {
            coo::sort_arrays(controls, _rows, _cols, _nnz);
        }

        if (!noDuplicates) {
            reduce_duplicates2(controls);
        }

    }


    matrix_coo::matrix_coo(Controls &controls,
                           index_type nrows,
                           index_type ncols,
                           index_type nnz,
                           cl::Buffer &rows,
                           cl::Buffer &cols,
                           bool sorted,
                           bool noDuplicates
    )
            : matrix_base(nrows, ncols, nnz), _rows(rows), _cols(cols) {

        if (!sorted) {
            coo::sort_arrays(controls, _rows, _cols, _nnz);
        }

        if (!noDuplicates) {
            reduce_duplicates2(controls);
        }
    }

    matrix_coo &matrix_coo::operator=(const matrix_coo &other) {
        _ncols = other._ncols;
        _nrows = other._nrows;
        _nnz = other._nnz;
        _rows = other._rows;
        _cols = other._cols;
        return *this;
    }

    void matrix_coo::reduce_duplicates(Controls &controls) {
        // ------------------------------------ prepare array to count positions ----------------------
        cl::Buffer positions;
        CHECK_CL(positions = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (_nnz + 1)),
                CLBOOL_CREATE_BUFFER_ERROR, 97987319);

        auto prepare_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("prepare_positions", "prepare_array_for_positions");
        prepare_positions.set_work_size(_nnz);
        CHECK_RUN(prepare_positions.run(controls, positions, _rows, _cols, _nnz).wait(), 367279701);

        // ------------------------------------ calculate positions, get new_size -----------------------------------
        uint32_t new_nnz;
        prefix_sum(controls, positions, new_nnz, _nnz + 1);

        cl::Buffer new_rows;
        CHECK_CL(new_rows = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_nnz),
        CLBOOL_CREATE_BUFFER_ERROR, 9739721);
        cl::Buffer new_cols;
        CHECK_CL(new_cols = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_nnz),
        CLBOOL_CREATE_BUFFER_ERROR, 8002871);

        auto set_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>
                ("set_positions", "set_positions");
        set_positions.set_work_size(_nnz);
        CHECK_RUN(set_positions.run(controls, new_rows, new_cols, _rows, _cols, positions, _nnz).wait(), 848276172);
        _rows = std::move(new_rows);
        _cols = std::move(new_cols);
        _nnz = new_nnz;
    }

    void matrix_coo::reduce_duplicates2(Controls &controls) {
        // ------------------------------------ prepare array to count positions ----------------------

        uint32_t groups_num = (_nnz + controls.max_wg_size - 1) / controls.max_wg_size;
        cl::Buffer duplicates_per_tb;
        CHECK_CREATE_BUF(duplicates_per_tb = utils::create_buffer(controls, groups_num + 1), 87660101);

        auto init_duplicates = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>
                ("coo_reduce_duplicates", "init_duplicates");
        init_duplicates.set_block_size(controls.max_wg_size);
        init_duplicates.set_work_size(groups_num);

        TIMEIT("init_duplicates run in: ",
        CHECK_RUN(init_duplicates.run(controls, _rows, _cols, duplicates_per_tb, _nnz, groups_num), 56575111)
        );

        auto reduce_tb = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("coo_reduce_duplicates", "reduce_duplicates_tb");
        reduce_tb.set_block_size(controls.max_wg_size);
        reduce_tb.set_work_size(_nnz);

        TIMEIT("reduce_tb run in: ",
        CHECK_RUN(reduce_tb.run(controls, _rows, _cols, duplicates_per_tb, _nnz), 98666151)
        );

        uint32_t total_duplicates;
        uint32_t new_nnz;
        TIMEIT("prefix_sum run in: ", prefix_sum(controls, duplicates_per_tb, total_duplicates, groups_num + 1));

        if (total_duplicates == 0) return;

        new_nnz = _nnz - total_duplicates;

        cl::Buffer new_rows;
        CHECK_CREATE_BUF(new_rows = utils::create_buffer(controls, new_nnz), 8777621);

        cl::Buffer new_cols;
        CHECK_CREATE_BUF(new_cols = utils::create_buffer(controls, new_nnz), 815134914);

        auto shift_tb = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("coo_reduce_duplicates", "shift_tb");
        shift_tb.set_block_size(controls.max_wg_size);
        shift_tb.set_work_size(_nnz);

        TIMEIT("shift_tb tun in ",
        CHECK_RUN(shift_tb.run(controls, _rows, _cols, new_rows, new_cols, duplicates_per_tb, _nnz), 102552366)
        );

        _nnz = new_nnz;
        _rows = std::move(new_rows);
        _cols = std::move(new_cols);
    }
}
