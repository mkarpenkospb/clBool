#include "clBool_tests.hpp"

#include "coo.hpp"

using namespace clbool;
using namespace clbool::coo_utils;
using namespace clbool::utils;


bool test_multiplication_merge(Controls &controls, uint32_t size, uint32_t k) {
    SET_TIMER
    std::cout << "----------------------- size = " << size << ", k = " << k << "----------------------\n";

    uint32_t max_size = size;
    uint32_t nnz_max = max_size * k;

    matrix_dcsr_cpu a_cpu = coo_pairs_to_dcsr_cpu(generate_coo_pairs_cpu(nnz_max, max_size));
    matrix_dcsr_cpu c_cpu;

    {
        START_TIMING
        matrix_multiplication_cpu(c_cpu, a_cpu, a_cpu);
        END_TIMING("matrix multiplication on CPU: ")
    }

    std::cout << "final matrix size: " << c_cpu.cols().size() << std::endl;

    matrix_dcsr a_gpu = matrix_dcsr_from_cpu(controls, a_cpu, max_size);
    matrix_dcsr c_gpu;

    std::cout << "matrix size is "  << a_gpu.nnz() << std::endl;

    {
        START_TIMING
        dcsr::matrix_multiplication(controls, c_gpu, a_gpu, a_gpu);
        END_TIMING("matrix multiplication on DEVICE: ")
    }

    return compare_matrices(controls, c_gpu, c_cpu);
}


bool test_multiplication_hash(Controls &controls, uint32_t size, uint32_t k) {

    SET_TIMER

    LOG << "\n\nITER ------------------------ size = " << size << ", k = " << k << "-----------------------------\n";

    uint32_t max_size = size;
    uint32_t nnz_max = max_size * k;
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
        dcsr::matrix_multiplication_hash(controls, c_gpu, a_gpu, a_gpu);
        END_TIMING("matrix multiplication on DEVICE: ")
    }

    return compare_matrices(controls, c_gpu, c_cpu);
}


bool test_reduce(Controls &controls, uint32_t size, uint32_t k) {
    SET_TIMER

    uint32_t max_size = size;
    uint32_t nnz_max = max_size * k;

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
        dcsr::reduce(controls, a_gpu, a_gpu);
        END_TIMING("reduce on DEVICE: ")
    }

    return compare_matrices(controls, a_gpu, a_cpu);
}


bool test_submatrix(Controls &controls, uint32_t size, uint32_t k) {
    SET_TIMER

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};

    uint32_t max_size = size;
    uint32_t nnz_max = max_size * k;

    LOG << " ------------------------------- k = " << k << ", size = " << size
        << "-------------------------------------------\n"
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
        dcsr::submatrix(controls, c_gpu, a_gpu, i, j, nrows, ncols);
        END_TIMING("submatrix on DEVICE: ")
    }

    return compare_matrices(controls, c_gpu, c_cpu);
}


bool test_transpose(Controls &controls, uint32_t size, uint32_t k) {
    SET_TIMER

    uint32_t max_size = size;
    uint32_t nnz_max = max_size * k;

    LOG << " ------------------------------- k = " << k << ", size = " << size
        << " -------------------------------------------\n"
        << " -------------------------------"
        << "max_size = " << size << ", nnz_max = " << nnz_max
        << " -------------------------------";

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
        dcsr::transpose(controls, a_gpu, a_gpu);
        END_TIMING("transpose: ")
    }

    return compare_matrices(controls, a_gpu, a_dcsr_cpu_tr);
}


bool test_addition_coo(Controls &controls, uint32_t size, uint32_t k_a, uint32_t k_b) {
    SET_TIMER

    std::cout << " ------------------------------- k_a = " << k_a << ", k_b = " << k_b << ", "  << "size = " << size
    << " -------------------------------------------\n";


    matrix_coo_cpu_pairs matrix_res_cpu;
    matrix_coo_cpu_pairs matrix_a_cpu = coo_utils::generate_coo_pairs_cpu(size * k_a, size);
    matrix_coo_cpu_pairs matrix_b_cpu = coo_utils::generate_coo_pairs_cpu(size * k_b, size);

    matrix_coo matrix_res_gpu;
    matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu, size, size);
    matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu, size, size);

    {
        START_TIMING
        coo_utils::matrix_addition_cpu(matrix_res_cpu, matrix_a_cpu, matrix_b_cpu);
        END_TIMING("matrix_addition on CPU: ")
    }

    {
        START_TIMING
        coo::matrix_addition(controls, matrix_res_gpu, matrix_a_gpu, matrix_b_gpu);
        END_TIMING("matrix_addition on DEVICE: ")
    }

    cpu_buffer rows_cpu;
    cpu_buffer cols_cpu;

    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, matrix_res_cpu);

    if (matrix_res_gpu.nnz() != rows_cpu.size()) {
        std::cerr << "diff nnz, gpu: " << matrix_res_gpu.nnz() << " vs cpu: " << rows_cpu.size() << std::endl;
        return false;
    }

    return compare_buffers(controls, matrix_res_gpu.rows_gpu(), rows_cpu, rows_cpu.size(), "rows") &&
           compare_buffers(controls, matrix_res_gpu.cols_gpu(), cols_cpu, cols_cpu.size(), "cols");
}

bool test_kronecker_coo(clbool::Controls &controls,
                        uint32_t size_a, uint32_t size_b, uint32_t nnz_a, uint32_t nnz_b, uint32_t k) {
    SET_TIMER

    LOG << " ---------------------------------- k = " <<
    k << ", size_a = " << size_a << ", size_b = " << size_b
    << " ----------------------------------\n"
    << " ---------------------------------- " <<
    "nnz_a = " << nnz_a << ", nnz_b = " << nnz_b << ", result size: " << nnz_a * nnz_b <<
    " ----------------------------------\n";

    matrix_coo_cpu_pairs matrix_res_cpu;
    matrix_coo_cpu_pairs matrix_a_cpu = coo_utils::generate_coo_pairs_cpu(nnz_a, size_a);

    matrix_coo matrix_res_gpu;
    matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu, size_a, size_a);

    {
        START_TIMING
        kronecker_product_cpu(matrix_res_cpu, matrix_a_cpu, matrix_a_cpu, size_b, size_b);
        END_TIMING("kronecker product on CPU: ")
    }

    {
        START_TIMING
        coo::kronecker_product(controls, matrix_res_gpu, matrix_a_gpu, matrix_a_gpu);
        END_TIMING("kronecker product on DEVICE: ")
    }

    cpu_buffer rows_cpu;
    cpu_buffer cols_cpu;

    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, matrix_res_cpu);

    return
//    compare_buffers(controls, matrix_res_gpu.rows_gpu(), rows_cpu, rows_cpu.size(), "rows") &&
           compare_buffers(controls, matrix_res_gpu.cols_gpu(), cols_cpu, cols_cpu.size(), "cols");

}

bool test_kronecker_dcsr(clbool::Controls &controls,
                         uint32_t size_a, uint32_t size_b, uint32_t nnz_a, uint32_t nnz_b, uint32_t k) {
    SET_TIMER

    LOG << " ---------------------------------- k = " <<
    k << ", size_a = " << size_a << ", size_b = " << size_b
    << " ----------------------------------\n"
    << " ---------------------------------- " <<
    "nnz_a = " << nnz_a << ", nnz_b = " << nnz_b << ", result size: " << nnz_a * nnz_b <<
    " ----------------------------------\n";

    matrix_dcsr_cpu matrix_res_cpu;
    matrix_dcsr matrix_res_gpu;
    matrix_dcsr matrix_a_gpu;

    {
        matrix_coo_cpu_pairs matrix_coo_res_cpu;
        matrix_coo_cpu_pairs matrix_coo_a_cpu = coo_utils::generate_coo_pairs_cpu(nnz_a, size_a);

        matrix_a_gpu = matrix_dcsr_from_cpu(controls,
                                            coo_utils::coo_pairs_to_dcsr_cpu(matrix_coo_a_cpu), size_a);

        {
            START_TIMING
            kronecker_product_cpu(matrix_coo_res_cpu, matrix_coo_a_cpu, matrix_coo_a_cpu, size_b, size_b);
            END_TIMING("kronecker product on CPU: ")
        }

        matrix_res_cpu = coo_utils::coo_pairs_to_dcsr_cpu(matrix_coo_res_cpu);
    }

    {
        START_TIMING
        dcsr::kronecker_product(controls, matrix_res_gpu, matrix_a_gpu, matrix_a_gpu);
        END_TIMING("kronecker product on DEVICE: ")
    }

    return compare_matrices(controls, matrix_res_gpu, matrix_res_cpu);
}

bool test_reduce_duplicates(Controls &controls) {
    uint32_t size = 10374663;

    // -------------------- create indices ----------------------------

    cpu_buffer rows_cpu(size);
    cpu_buffer cols_cpu(size);

    std::vector<uint32_t> rows_from_gpu(size);
    std::vector<uint32_t> cols_from_gpu(size);

    coo_utils::fill_random_matrix(rows_cpu, cols_cpu, 1043);

    // -------------------- create and sort cpu matrix ----------------------------
    matrix_coo_cpu_pairs m_cpu;
    coo_utils::form_cpu_matrix(m_cpu, rows_cpu, cols_cpu);
    std::sort(m_cpu.begin(), m_cpu.end());

    // -------------------- create and sort gpu buffers ----------------------------

    cl::Buffer rows_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
    cl::Buffer cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);

    controls.queue.enqueueWriteBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, rows_cpu.data());
    controls.queue.enqueueWriteBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, cols_cpu.data());

    coo::sort_arrays(controls, rows_gpu, cols_gpu, size);

    // ------------------ now reduce cpu matrix and read result in vectors ------------------------

    std::cout << "\nmatrix cpu before size: " << m_cpu.size() << std::endl;
    m_cpu.erase(std::unique(m_cpu.begin(), m_cpu.end()), m_cpu.end());
    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, m_cpu);
    std::cout << "\nmatrix cpu after size: " << m_cpu.size() << std::endl;

    // ------------------ now reduce gpu buffers and read in vectors ------------------------
    uint32_t new_size;
//    coo::reduce_duplicates(controls, rows_gpu, cols_gpu, reinterpret_cast<uint32_t &>(new_size), size);

    rows_from_gpu.resize(new_size);
    cols_from_gpu.resize(new_size);

    controls.queue.enqueueReadBuffer(rows_gpu, CL_TRUE, 0, sizeof(uint32_t) * new_size, rows_from_gpu.data());
    controls.queue.enqueueReadBuffer(cols_gpu, CL_TRUE, 0, sizeof(uint32_t) * new_size, cols_from_gpu.data());

    if (rows_from_gpu == rows_cpu && cols_from_gpu == cols_cpu) {
        std::cout << "correct reduce" << std::endl;
    } else {
        std::cerr << "incorrect reduce" << std::endl;
    }

    return true;
}