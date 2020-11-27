#include "matrix_coo.hpp"

matrix_coo::matrix_coo(Controls &controls,
                       uint32_t nRows,
                       uint32_t nCols,
                       uint32_t nEntities)
        : matrix_base(nRows, nCols, nEntities),
          _rows_indices_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities)),
          _cols_indices_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities)),
          _rows_indices_cpu(std::vector<uint32_t>(0, n_entities)),
          _cols_indices_cpu(std::vector<uint32_t>(0, n_entities)) {}


matrix_coo::matrix_coo(Controls &controls,
                       uint32_t nRows,
                       uint32_t nCols,
                       uint32_t nEntities,
                       std::vector<uint32_t> rows_indices,
                       std::vector<uint32_t> cols_indices,
                       bool sorted)
        : matrix_base(nRows, nCols, nEntities),
          _rows_indices_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities)),
          _cols_indices_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities)),
          _rows_indices_cpu(std::move(rows_indices)), _cols_indices_cpu(std::move(cols_indices)) {
    try {

        controls.queue.enqueueWriteBuffer(_rows_indices_gpu, CL_TRUE, 0, sizeof(uint32_t) * _rows_indices_cpu.size(),
                                          _rows_indices_cpu.data());

        controls.queue.enqueueWriteBuffer(_cols_indices_gpu, CL_TRUE, 0, sizeof(uint32_t) * _cols_indices_cpu.size(),
                                          _cols_indices_cpu.data());

        if (!sorted) {
            sort_arrays(controls, _rows_indices_gpu, _cols_indices_gpu, n_entities);


            controls.queue.enqueueReadBuffer(_rows_indices_gpu, CL_TRUE, 0, sizeof(uint32_t) * _rows_indices_cpu.size(),
                                             _rows_indices_cpu.data());

            controls.queue.enqueueReadBuffer(_cols_indices_gpu, CL_TRUE, 0, sizeof(uint32_t) * _cols_indices_cpu.size(),
                                             _cols_indices_cpu.data());
        }

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        throw std::runtime_error(exception.str());
    }
}


matrix_coo::matrix_coo(Controls &controls,
                              uint32_t nRows,
                              uint32_t nCols,
                              uint32_t nEntities,
                              cl::Buffer rows,
                              cl::Buffer cols)
        : matrix_base(nRows, nCols, nEntities),
          _rows_indices_gpu(std::move(rows)),
          _cols_indices_gpu(std::move(cols)),
          _rows_indices_cpu(std::vector<uint32_t>(nEntities)),
          _cols_indices_cpu(std::vector<uint32_t>(nEntities)) {
    try {
        controls.queue.enqueueReadBuffer(_rows_indices_gpu, CL_TRUE, 0, sizeof(uint32_t) * n_entities,
                                         _rows_indices_cpu.data());

        controls.queue.enqueueReadBuffer(_cols_indices_gpu, CL_TRUE, 0, sizeof(uint32_t) * n_entities,
                                         _cols_indices_cpu.data());
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        throw std::runtime_error(exception.str());
    }
}

matrix_coo &matrix_coo::operator=(matrix_coo other) {
    n_cols = other.n_cols;
    n_rows = other.n_rows;
    n_entities = other.n_entities;
    _rows_indices_gpu = std::move(other._rows_indices_gpu);
    _cols_indices_gpu = std::move(other._cols_indices_gpu);
    _rows_indices_cpu = std::move(other._rows_indices_cpu);
    _cols_indices_cpu = std::move(other._cols_indices_cpu);
    return *this;
}


