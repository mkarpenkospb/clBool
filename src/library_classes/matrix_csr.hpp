#pragma once

#include "matrix_base.hpp"
#include "controls.hpp"

class matrix_csr : public details::matrix_base<CSR> {
private:
    bool abstract() override {
        return false;
    };

    cl::Buffer rows_pointers_gpu;
    cl::Buffer cols_indexes_gpu;

    std::vector<uint32_t> rows_pointers_cpu;
    std::vector<uint32_t> cols_indexes_cpu;

public:

    matrix_csr(Controls controls, uint32_t nRows, uint32_t nCols, uint32_t nEntities)
            : matrix_base(nRows, nCols, nEntities)
            // TODO: confirm if we need CL_MEM_READ_WRITE
            , rows_pointers_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
            , cols_indexes_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
            , rows_pointers_cpu(std::vector<uint32_t> (0, n_entities))
            , cols_indexes_cpu(std::vector<uint32_t> (0, n_entities))
    {}


    matrix_csr(Controls controls, uint32_t nRows, uint32_t nCols, uint32_t nEntities,
               std::vector<uint32_t> rows_indexes, std::vector<uint32_t> cols_indexes)
            : matrix_base(nRows, nCols, nEntities)
            , rows_pointers_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
            , cols_indexes_gpu(cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n_entities))
            , rows_pointers_cpu(std::move(rows_indexes))
            , cols_indexes_cpu(std::move(cols_indexes))
    {
        try {
            controls.queue.enqueueWriteBuffer(rows_pointers_gpu, CL_TRUE, 0, sizeof(uint32_t) * rows_pointers_cpu.size(),
                                              rows_pointers_cpu.data());

            controls.queue.enqueueWriteBuffer(cols_indexes_gpu, CL_TRUE, 0, sizeof(uint32_t) * cols_indexes_cpu.size(),
                                              cols_indexes_cpu.data());

        } catch (const cl::Error& e) {
            std::stringstream exception;
            exception << "\n" << e.what() << " : " << e.err() << "\n";
            throw std::runtime_error(exception.str());
        }
    }

    const auto& get_rows_pointers_cpu() const {
        return rows_pointers_cpu;
    }

    const auto& get_column_indexes_cpu() const {
        return cols_indexes_cpu;
    }

};