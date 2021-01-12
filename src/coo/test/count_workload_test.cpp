#include "coo_tests.hpp"


#include "coo_tests.hpp"
#include "../../cl_includes.hpp"
#include "../../library_classes/controls.hpp"
#include "../../utils.hpp"
#include "../coo_utils.hpp"
#include "../coo_matrix_multiplication.hpp"

using coo_utils::matrix_coo_cpu;




void testCountWorkload() {
    Controls controls = utils::create_controls();

    uint32_t nnz_limit = 1'000'000;
    uint32_t max_size = 1'024;
    matrix_coo_cpu matrix_a_cpu = coo_utils::generate_random_matrix_cpu(nnz_limit, max_size);
    matrix_coo_cpu matrix_b_cpu = coo_utils::generate_random_matrix_cpu(nnz_limit + 1, max_size + 1);


    cpu_buffer a_rows_pointers_cpu;
    cpu_buffer a_rows_compressed_cpu;
    coo_utils::get_rows_pointers_and_compressed(a_rows_pointers_cpu, a_rows_compressed_cpu, matrix_a_cpu);


    cpu_buffer b_rows_pointers_cpu;
    cpu_buffer b_rows_compressed_cpu;
    coo_utils::get_rows_pointers_and_compressed(b_rows_pointers_cpu, b_rows_compressed_cpu, matrix_b_cpu);

//    coo_utils::print_matrix(matrix_a_cpu);
//    coo_utils::print_matrix(matrix_b_cpu);

    matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu);
    matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu);

    cl::Buffer a_rows_pointers;
    cl::Buffer a_rows_compressed;
    uint32_t a_nzr;

    cl::Buffer b_rows_pointers;
    cl::Buffer b_rows_compressed;
    uint32_t b_nzr;


//    cpu_buffer rows;
//    cpu_buffer cols;
//    coo_utils::get_vectors_from_cpu_matrix(rows, cols, matrix_b_cpu);
//    utils::compare_buffers(controls, matrix_b_gpu.rows_indices_gpu(), rows, matrix_b_cpu.size());
//    utils::compare_buffers(controls, matrix_b_gpu.cols_indices_gpu(), cols, matrix_b_cpu.size());

    create_rows_pointers(controls, a_rows_pointers, a_rows_compressed, matrix_a_gpu.rows_indices_gpu(),
                         matrix_a_gpu.nnz(), a_nzr);

    create_rows_pointers(controls, b_rows_pointers, b_rows_compressed, matrix_b_gpu.rows_indices_gpu(),
                         matrix_b_gpu.nnz(), b_nzr);

//    utils::compare_buffers(controls, a_rows_pointers, a_rows_pointers_cpu, a_rows_pointers_cpu.size());
//    utils::compare_buffers(controls, b_rows_pointers, b_rows_pointers_cpu, b_rows_pointers_cpu.size());
//    utils::compare_buffers(controls, a_rows_compressed, a_rows_compressed_cpu, a_rows_compressed_cpu.size());
//    utils::compare_buffers(controls, b_rows_compressed, b_rows_compressed_cpu, b_rows_compressed_cpu.size());


    cl::Buffer workload;
    count_workload(controls, workload,
                   a_rows_pointers, matrix_a_gpu.cols_indices_gpu(),
                   b_rows_compressed, b_rows_pointers, matrix_b_gpu.cols_indices_gpu(),
                   a_nzr, b_nzr);

    std::cout << "finish gpu counting" << std::endl;

    //--------------------------------- cpu part -----------------------------------
    cpu_buffer workload_cpu(a_nzr);



    coo_utils::get_workload(workload_cpu, matrix_a_cpu, a_rows_pointers_cpu,
                            b_rows_pointers_cpu, b_rows_compressed_cpu, a_nzr);

    std::cout << "finish cpu counting" << std::endl;

    utils::compare_buffers(controls, workload, workload_cpu, a_nzr);
}

