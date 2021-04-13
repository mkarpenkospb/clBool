#pragma once

using coordinates = std::pair<uint32_t, uint32_t>;
using matrix_coo_cpu_pairs = std::vector<coordinates>;
using cpu_buffer = std::vector<uint32_t>;

class matrix_dcsr_cpu {
    cpu_buffer _rpt;
    cpu_buffer _rows;
    cpu_buffer _cols;

public:
    matrix_dcsr_cpu(cpu_buffer rpt, cpu_buffer rows, cpu_buffer cols)
            : _rpt(std::move(rpt)), _rows(std::move(rows)),
              _cols(std::move(cols)) {}

    matrix_dcsr_cpu() = default;

    matrix_dcsr_cpu &operator=(matrix_dcsr_cpu other) {
        _rpt = std::move(other._rpt);
        _rows = std::move(other._rows);
        _cols = std::move(other._cols);
        return *this;
    }

    cpu_buffer &rpt() {
        return _rpt;
    }

    cpu_buffer &rows() {
        return _rows;
    }

    cpu_buffer &cols() {
        return _cols;
    }

    const cpu_buffer &rpt() const {
        return _rpt;
    }

    const cpu_buffer &rows() const {
        return _rows;
    }

    const cpu_buffer &cols() const {
        return _cols;
    }

};


class matrix_coo_cpu {
    cpu_buffer _rows_indices;
    cpu_buffer _cols_indices;

public:
    matrix_coo_cpu(cpu_buffer rows_indices, cpu_buffer cols_indices)
            : _rows_indices(std::move(rows_indices))
            , _cols_indices(std::move(cols_indices))
            {}

    matrix_coo_cpu() = default;

    matrix_coo_cpu &operator=(matrix_coo_cpu other) {
        _rows_indices = std::move(other._rows_indices);
        _cols_indices = std::move(other._cols_indices);
        return *this;
    }

    cpu_buffer &rows_indices() {
        return _rows_indices;
    }

    cpu_buffer &cols_indices() {
        return _cols_indices;
    }

    const cpu_buffer &rows_indices() const {
        return _rows_indices;
    }

    const cpu_buffer &cols_indices() const {
        return _cols_indices;
    }

};