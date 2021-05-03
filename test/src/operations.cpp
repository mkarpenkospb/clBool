#include "clBool_tests.hpp"

#include "coo.hpp"

using namespace clbool;
using namespace clbool::coo_utils;
using namespace clbool::utils;


bool test_multiplication_merge(Controls controls, uint32_t size, uint32_t k) {
    SET_TIMER

    uint32_t max_size = size;
    uint32_t nnz_max = std::max(10u, max_size * k);

    matrix_dcsr_cpu a_cpu = coo_pairs_to_dcsr_cpu(generate_coo_pairs_cpu(nnz_max, max_size));
    matrix_dcsr_cpu c_cpu;

    {
        START_TIMING
        matrix_multiplication_cpu(c_cpu, a_cpu, a_cpu);
        END_TIMING("matrix multiplication on CPU: ")
    }

    std::cout << "matrix_multiplication_cpu finished" << std::endl;

    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
    matrix_dcsr c_gpu;


    {
        START_TIMING
        matrix_multiplication(controls, c_gpu, a_gpu, a_gpu);
        END_TIMING("matrix multiplication on DEVICE: ")
    }

    return compare_matrices(controls, c_gpu, c_cpu);
}


bool test_multiplication_hash(Controls controls, uint32_t size, uint32_t k) {


    SET_TIMER

    LOG << "\n\nITER ------------------------ size = " << size << ", k = " << k << "-----------------------------\n";

    uint32_t max_size = size;
    uint32_t nnz_max = std::max(10u, max_size * k);
    matrix_dcsr_cpu a_cpu;
    {
        START_TIMING
        a_cpu = coo_pairs_to_dcsr_cpu(generate_coo_pairs_cpu(nnz_max, max_size));
        END_TIMING("a_cpu created: ")
    }


    matrix_dcsr_cpu c_cpu;
    {
        START_TIMING
        matrix_multiplication_cpu(c_cpu, a_cpu, a_cpu);
        END_TIMING("matrix multiplication on CPU: ")
    }

    matrix_dcsr a_gpu;
    {
        START_TIMING
        a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
        END_TIMING("matrix_dcsr_from_cpu: ")
    }


    matrix_dcsr c_gpu;
    {
        START_TIMING
        matrix_multiplication_hash(controls, c_gpu, a_gpu, a_gpu);
        END_TIMING("matrix multiplication on DEVICE: ")
    }

    compare_matrices(controls, c_gpu, c_cpu);
}


bool test_reduce(Controls controls, uint32_t size, uint32_t k) {
    SET_TIMER

    uint32_t max_size = size;
    uint32_t nnz_max = std::max(10u, max_size * k);

    LOG << " ------------------------------- k = " << k << ", size = " << size
              << " -------------------------------------------\n"
              << "max_size = " << size << ", nnz_max = " << nnz_max;

    matrix_dcsr_cpu a_cpu = coo_to_dcsr_cpu(generate_coo_cpu(nnz_max, max_size));

    matrix_dcsr a_gpu;
    {
        START_TIMING
        a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
        END_TIMING("matrix_dcsr_from_cpu: ")
    }


    {
        START_TIMING
        utils::reduce(a_cpu, a_cpu);
        END_TIMING("reduce on CPU: ")
    }


    {
        START_TIMING
        reduce(controls, a_gpu, a_gpu);
        END_TIMING("reduce on DEVICE: ")
    }

    return compare_matrices(controls, a_gpu, a_cpu);
}


bool test_submatrix(Controls controls, uint32_t size, uint32_t k, uint32_t iter) {
    SET_TIMER

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};

    uint32_t max_size = size;
    uint32_t nnz_max = std::max(10u, max_size * k);

    LOG << " ------------------------------- k = " << k << ", size = " << size
              << ", iter = " << iter << "-------------------------------------------\n"
              << "max_size = " << size << ", nnz_max = " << nnz_max;

    std::uniform_int_distribution<uint32_t> rnd_idx{0, max_size};
    uint32_t i1 = rnd_idx(mersenne_engine);
    uint32_t i2 = rnd_idx(mersenne_engine);
    uint32_t i3 = rnd_idx(mersenne_engine);
    uint32_t i4 = rnd_idx(mersenne_engine);

    uint32_t i = std::min(i1, i2);
    uint32_t nrows = std::max(i1, i2) - i;

    uint32_t j = std::min(i3, i4);
    uint32_t ncols = std::max(i3, i4) - j;


    LOG << "i = " << i << ", j = " << j << ", nrows = " << nrows << ", ncols = " << ncols;

    matrix_dcsr_cpu a_cpu = coo_pairs_to_dcsr_cpu(generate_coo_pairs_cpu(nnz_max, max_size));
    matrix_dcsr_cpu c_cpu;


    {
        START_TIMING
        submatrix_cpu(c_cpu, a_cpu, i, j, nrows, ncols);
        END_TIMING("submatrix on CPU: ")
    }

    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
    matrix_dcsr c_gpu;

    {
        START_TIMING
        submatrix(controls, c_gpu, a_gpu, i, j, nrows, ncols);
        END_TIMING("submatrix on DEVICE: ")
    }

    return compare_matrices(controls, c_gpu, c_cpu);
}


bool test_transpose(Controls controls, uint32_t size, uint32_t k) {
    SET_TIMER

    uint32_t max_size = size;
    uint32_t nnz_max = std::max(10u, max_size * k);

    LOG << " ------------------------------- k = " << k << ", size = " << size
    << " -------------------------------------------\n"
    << "max_size = " << size << ", nnz_max = " << nnz_max;

    matrix_coo_cpu a_coo_cpu = generate_coo_cpu(nnz_max, max_size);
    matrix_dcsr_cpu a_dcsr_cpu = coo_to_dcsr_cpu(a_coo_cpu);
    a_coo_cpu.transpose();
    matrix_dcsr_cpu a_dcsr_cpu_tr = coo_to_dcsr_cpu(a_coo_cpu);

    matrix_dcsr a_gpu;
    {
        START_TIMING
        a_gpu = matrix_dcsr_from_cpu(controls, a_dcsr_cpu, max_size);
        END_TIMING("matrix_dcsr_from_cpu: ")
    }

    {
        START_TIMING
        transpose(controls, a_gpu, a_gpu);
        END_TIMING("transpose: ")
    }

    return compare_matrices(controls, a_gpu, a_dcsr_cpu_tr);
}


bool test_addition_coo(Controls controls, uint32_t size_a, uint32_t size_b, uint32_t k_a, uint32_t k_b) {
    SET_TIMER

    matrix_coo_cpu_pairs matrix_res_cpu;
    matrix_coo_cpu_pairs matrix_a_cpu = coo_utils::generate_coo_pairs_cpu(size_a * k_a, size_a);
    matrix_coo_cpu_pairs matrix_b_cpu = coo_utils::generate_coo_pairs_cpu(size_b * k_b, size_b);

    matrix_coo matrix_res_gpu;
    matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu);
    matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu);

    coo_utils::matrix_addition_cpu(matrix_res_cpu, matrix_a_cpu, matrix_b_cpu);

    matrix_addition(controls, matrix_res_gpu, matrix_a_gpu, matrix_b_gpu);

    std::vector<uint32_t> rows_cpu;
    std::vector<uint32_t> cols_cpu;

    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, matrix_res_cpu);

    return compare_buffers(controls, matrix_res_gpu.rows_gpu(), rows_cpu, rows_cpu.size()) &&
           compare_buffers(controls, matrix_res_gpu.cols_gpu(), cols_cpu, cols_cpu.size());
}
