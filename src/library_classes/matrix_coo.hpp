#pragma once


#include "matrix_base.hpp"
#include "controls.hpp"
#include "../COO/coo_initialization.hpp"


#include <vector>

class matrix_coo : public details::matrix_base<COO> {
private:
    // buffers for uint32only;

    bool abstract() override {
        return false;
    };

    cl::Buffer _rows_indexes_gpu;
    cl::Buffer _cols_indexes_gpu;

    std::vector<uint32_t> _rows_indexes_cpu;
    std::vector<uint32_t> _cols_indexes_cpu;

public:

    matrix_coo(Controls controls, uint32_t nRows, uint32_t nCols, uint32_t nEntities)
    : matrix_base(nRows, nCols, nEntities)
    // TODO: confirm if we need CL_MEM_READ_WRITE
    , _rows_indexes_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
    , _cols_indexes_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
    , _rows_indexes_cpu(std::vector<uint32_t> (0, n_entities))
    , _cols_indexes_cpu(std::vector<uint32_t> (0, n_entities))
    {}


    matrix_coo(Controls controls, uint32_t nRows, uint32_t nCols, uint32_t nEntities,
               std::vector<uint32_t> rows_indexes, std::vector<uint32_t> cols_indexes, bool sorted = false)
    : matrix_base(nRows, nCols, nEntities)
    , _rows_indexes_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
    , _cols_indexes_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
    , _rows_indexes_cpu(std::move(rows_indexes))
    , _cols_indexes_cpu(std::move(cols_indexes))
    {
        try {

            controls.queue.enqueueWriteBuffer(_rows_indexes_gpu, CL_TRUE, 0, sizeof(uint32_t) * _rows_indexes_cpu.size(),
                                              _rows_indexes_cpu.data());

            controls.queue.enqueueWriteBuffer(_cols_indexes_gpu, CL_TRUE, 0, sizeof(uint32_t) * _cols_indexes_cpu.size(),
                                              _cols_indexes_cpu.data());

            if (!sorted) {
                sort_arrays(controls, _rows_indexes_gpu, _cols_indexes_gpu, n_entities);
            }

            controls.queue.enqueueReadBuffer(_rows_indexes_gpu, CL_TRUE, 0, sizeof(uint32_t) * _rows_indexes_cpu.size(),
                                             _rows_indexes_cpu.data());

            controls.queue.enqueueReadBuffer(_cols_indexes_gpu, CL_TRUE, 0, sizeof(uint32_t) * _cols_indexes_cpu.size(),
                                             _cols_indexes_cpu.data());

        } catch (const cl::Error& e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << e.err() << "\n";
        throw std::runtime_error(exception.str());
        }
    }

    const auto& get_rows_indexes_cpu() const {
        return _rows_indexes_cpu;
    }

    const auto& get_cols_indexes_cpu() const {
       return _cols_indexes_cpu;
    }

    const auto& get_rows_indexes_gpu() const {
        return _rows_indexes_gpu;
    }

    const auto& get_cols_indexes_gpu() const {
       return _cols_indexes_gpu;
    }

    const auto& get_rows_indexes_gpu() {
        return _rows_indexes_gpu;
    }

    const auto& get_cols_indexes_gpu() {
        return _cols_indexes_gpu;
    }


};
