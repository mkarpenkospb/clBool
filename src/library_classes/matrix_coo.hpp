#pragma once


#include "matrix_base.hpp"
#include "controls.hpp"
#include "../COO/coo_initialization.hpp"
#include "../utils.hpp"


#include <vector>

class matrix_coo : public details::matrix_base<COO> {
private:
    // buffers for uint32only;
    cl::Buffer _rows_indices_gpu;
    cl::Buffer _cols_indices_gpu;

    std::vector<uint32_t> _rows_indices_cpu;
    std::vector<uint32_t> _cols_indices_cpu;

public:

    // -------------------------------------- constructors -----------------------------

    matrix_coo() = default;

    matrix_coo(Controls &controls,
               uint32_t nRows,
               uint32_t nCols,
               uint32_t nEntities);

    matrix_coo(Controls &controls,
               uint32_t nRows,
               uint32_t nCols,
               uint32_t nEntities,
               std::vector<uint32_t> rows_indices,
               std::vector<uint32_t> cols_indices,
               bool sorted = false);

    /* we assume, that all input data are sorted */
    matrix_coo(Controls &controls,
               uint32_t nRows,
               uint32_t nCols,
               uint32_t nEntities,
               cl::Buffer rows,
               cl::Buffer cols);

    matrix_coo(matrix_coo const &other) = default;

    matrix_coo(matrix_coo &&other) noexcept = default;

    matrix_coo &operator=(matrix_coo other);

    const auto &rows_indices_cpu() const {
        return _rows_indices_cpu;
    }

    const auto &cols_indices_cpu() const {
        return _cols_indices_cpu;
    }

    const auto &rows_indices_gpu() const {
        return _rows_indices_gpu;
    }

    const auto &cols_indices_gpu() const {
        return _cols_indices_gpu;
    }

    const auto &rows_indices_gpu() {
        return _rows_indices_gpu;
    }

    const auto &cols_indices_gpu() {
        return _cols_indices_gpu;
    }

};

