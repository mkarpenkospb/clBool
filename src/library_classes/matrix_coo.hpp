#pragma once


#include "matrix_base.hpp"
#include "controls.hpp"
#include "../coo/coo_initialization.hpp"
#include "../common/utils.hpp"
#include <vector>

class matrix_coo : public details::matrix_base<COO> {
private:
    // buffers for uint32only;
    cl::Buffer _rows;
    cl::Buffer _cols;

public:

    // -------------------------------------- constructors -----------------------------

    matrix_coo() = default;

    matrix_coo(Controls &controls,
               index_type nrows,
               index_type ncols,
               index_type nnz);

    matrix_coo(index_type nrows,
               index_type ncols,
               index_type nnz,
               cl::Buffer &rows,
               cl::Buffer &cols
               );

    matrix_coo(Controls &controls,
               index_type nrows,
               index_type ncols,
               index_type nnz,
               std::vector<index_type> &rows_indices,
               std::vector<index_type> &cols_indices,
               bool sorted = false);

    /* we assume, that all input data are sorted */
    matrix_coo(Controls &controls,
               index_type nrows,
               index_type ncols,
               index_type nnz,
               cl::Buffer &rows,
               cl::Buffer &cols,
               bool sorted = false);

    matrix_coo(matrix_coo const &other) = default;

    matrix_coo(matrix_coo &&other) noexcept = default;

    matrix_coo &operator=(matrix_coo other);

    const auto &rows_gpu() const {
        return _rows;
    }

    const auto &cols_gpu() const {
        return _cols;
    }

    auto &rows_gpu() {
        return _rows;
    }

    auto &cols_gpu() {
        return _cols;
    }



};

