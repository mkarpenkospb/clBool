#pragma once


// format could be some global variable,
enum Format {
    COO,
    CSR
};



namespace details {

    template<Format format>
    class matrix_base {
    protected:

        Format sparse_format = format;

        uint32_t n_rows;
        uint32_t n_cols;
        uint32_t n_entities;

    public:

        matrix_base()
        : n_rows(0), n_cols(0), n_entities(0)
        {}

        matrix_base(uint32_t n_rows, uint32_t n_cols, uint32_t n_entities)
        : n_rows(n_rows), n_cols(n_cols), n_entities(n_entities)
        {}

        Format get_sparse_format() const {
            return sparse_format;
        };

        uint32_t nRows() const {
            return n_rows;
        };

        uint32_t nCols() const {
            return n_cols;
        };

        uint32_t nnz() const {
            return n_entities;
        };

    };
}