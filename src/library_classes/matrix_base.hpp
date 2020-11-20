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

        virtual bool abstract() = 0;

        Format sparse_format = format;

        uint32_t n_rows;
        uint32_t n_cols;
        uint32_t n_entities;

    public:
        // vot eto kostyl


        matrix_base(uint32_t n_rows, uint32_t n_cols, uint32_t n_entities)
            : n_rows(n_rows), n_cols(n_cols), n_entities(n_entities)
        {}

        Format get_sparse_format() const {
            return sparse_format;
        };

        uint32_t get_n_rows() const {
            return n_rows;
        };

        uint32_t get_n_cols() const {
            return n_cols;
        };

        uint32_t get_n_entities() const {
            return n_entities;
        };

    };
}