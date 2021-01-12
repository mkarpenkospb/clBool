#include <algorithm>
#include "coo_tests.hpp"
#include "../../cl_includes.hpp"
#include "../../library_classes/controls.hpp"
#include "../../utils.hpp"
#include "../coo_utils.hpp"
#include "../coo_matrix_multiplication.hpp"


using coo_utils::matrix_coo_cpu;
using cpu_buffer = std::vector<uint32_t>;

void print_matrix(const matrix_coo_cpu& m_cpu) {
    std::cout << std::endl;
    if (m_cpu.empty()) {
        std::cout << "empty matrix" << std::endl;
        return;
    }

    uint32_t curr_row = m_cpu.front().first;
    std::cout << "row " << curr_row << ": ";
    for (const auto &item: m_cpu) {
        if (item.first != curr_row) {
            curr_row = item.first;
            std::cout << std::endl;
            std::cout << "row " << curr_row << ": ";
        }
        std::cout << item.second << ", ";
    }
    std::cout << std::endl;
}


void get_rows_pointers_and_compressed(cpu_buffer &rows_pointers,
                                      cpu_buffer &rows_compressed,
                                      const matrix_coo_cpu &matrix_cpu) {
    if (matrix_cpu.empty()) return;

    size_t position = 0;
    uint32_t curr_row = matrix_cpu.front().first;
    rows_compressed.push_back(position);
    rows_pointers.push_back(position);
    for (const auto &item: matrix_cpu) {
        position ++;
        if (item.first != curr_row) {
            curr_row = item.first;
            rows_compressed.push_back(curr_row);
            rows_pointers.push_back(position);
        }
        std::cout << item.second << ", ";
    }
    std::cout << std::endl;

}


void testCOOtoDCSR() {
    Controls controls = utils::create_controls();
    // ----------------------------------------- create matrices ----------------------------------------

    uint32_t size = 15;
    matrix_coo_cpu m_cpu = coo_utils::generate_random_matrix_cpu(size, 10);
    cpu_buffer rows_pointers_cpu;
    cpu_buffer rows_compressed_cpu;
    get_rows_pointers_and_compressed(rows_pointers_cpu, rows_compressed_cpu, m_cpu);
    print_matrix(m_cpu);

//    std::cout << "rows_pointers_cpu:\n";
//    utils::print_cpu_buffer(rows_pointers_cpu);
//    std::cout << "rows_compressed_cpu:\n";
//    utils::print_cpu_buffer(rows_compressed_cpu);


    matrix_coo matrix_gpu = coo_utils::matrix_coo_from_cpu(controls, m_cpu);


    cl::Buffer rows_pointers;
    cl::Buffer rows_compressed;

    uint32_t a_nzr;

    create_rows_pointers(controls, rows_pointers, rows_compressed, matrix_gpu.rows_indices_gpu(), matrix_gpu.nnz(), a_nzr);

//    utils::
//    std::cout << "rows: " << std::endl;
//    utils::print_gpu_buffer(controls, matrix_gpu.rows_indices_gpu(), matrix_gpu.nnz());
//    std::cout << "cols: " << std::endl;
//    utils::print_gpu_buffer(controls, matrix_gpu.cols_indices_gpu(), matrix_gpu.nnz());


//    std::cout << "rows_pointers: " << std::endl;
//    utils::print_gpu_buffer(controls, rows_pointers, a_nzr + 1);
//    std::cout << "rows_compressed: " << std::endl;
//    utils::print_gpu_buffer(controls, rows_compressed, a_nzr);

}