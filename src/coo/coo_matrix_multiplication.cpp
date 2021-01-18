#include <numeric>
#include "coo_matrix_multiplication.hpp"
#include "coo_matrix_addition.hpp"
#include "../library_classes/controls.hpp"
#include "../utils.hpp"
#include "coo_utils.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_dcsr.hpp"

const uint32_t BINS_NUM = 38;
const uint32_t HEAP_MERGE_BLOCK_SIZE = 32;
typedef std::vector<uint32_t> cpu_buffer;

matrix_dcsr coo_to_dcsr_gpu(Controls &controls, const matrix_coo &m) {
    cl::Buffer rows_pointers;
    cl::Buffer rows_compressed;
    uint32_t nzr;
    create_rows_pointers(controls, rows_pointers, rows_compressed, m.rows_indices_gpu(), m.nnz(), nzr);

    return matrix_dcsr(rows_pointers, rows_compressed, m.rows_indices_gpu(),
                       m.nRows(), m.nCols(), m.nnz(), nzr
    );
}


/*
 * group_length - сколько выделить потоков всего
 * nnz_estimation - размер пирамиды в кернеле
 */
auto get_heap_kernel(Controls &controls,
                     uint32_t group_length,
                     const unsigned int nnz_estimation
) {
    cl::Program program;
    try {

        program = controls.create_program_from_file("../src/coo/cl/heap_merge.cl");
        uint32_t block_size = HEAP_MERGE_BLOCK_SIZE;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size << " -D NNZ_ESTIMATION=" << nnz_estimation;
        program.build(options.str().c_str());


        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, group_length);


        cl::Kernel heap_merge_kernel(program, "heap_merge");

        using KernelType = cl::KernelFunctor<cl::Buffer, uint32_t, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t>;

        KernelType heap_merge(heap_merge_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        return std::pair<KernelType, cl::EnqueueArgs>(heap_merge, eargs);
//        heap_merge(eargs, workload, a_rows_pointers, a_cols, b_rows_compressed, b_rows_pointers, a_nzr, b_nzr);
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }

}

auto get_copy_one_value_kernel(Controls &controls,
                               uint32_t group_length) {
    cl::Program program;
    try {

        program = controls.create_program_from_file("../src/coo/cl/copy_one_value.cl");
        uint32_t block_size = std::min(controls.block_size, std::min(32u, utils::ceil_to_power2(group_length)));

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, group_length);

        cl::Kernel copy_one_value_kernel(program, "copy_one_value");

        using KernelType = cl::KernelFunctor<cl::Buffer, uint32_t, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t>;

        KernelType copy_one_value(copy_one_value_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        return std::pair<KernelType, cl::EnqueueArgs>(copy_one_value, eargs);
//        heap_merge(eargs, workload, a_rows_pointers, a_cols, b_rows_compressed, b_rows_pointers, a_nzr, b_nzr);
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}

auto get_to_result_matrix_single_thread(Controls &controls,
                                        uint32_t group_length) {
    cl::Program program;
    try {
        program = controls.create_program_from_file("../src/coo/cl/to_result_matrix_single_thread.cl");
        uint32_t block_size = std::min(controls.block_size, std::min(32u, utils::ceil_to_power2(group_length)));

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, group_length);

        cl::Kernel to_result_kernel(program, "to_result");

        using KernelType = cl::KernelFunctor<cl::Buffer, uint32_t, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>;

        KernelType to_result(to_result_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        return std::pair<KernelType, cl::EnqueueArgs>(to_result_kernel, eargs);
//        heap_merge(eargs, workload, a_rows_pointers, a_cols, b_rows_compressed, b_rows_pointers, a_nzr, b_nzr);
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}

auto get_to_result_matrix_work_group(Controls &controls,
                                     uint32_t group_length) {
    cl::Program program;
    try {
        program = controls.create_program_from_file("../src/coo/cl/to_result_matrix_work_group.cl");
        // TODO: этот размер блока можно менять и смотреть, как будет быстрее
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = block_size * group_length;

        cl::Kernel to_result_kernel(program, "to_result");

        using KernelType = cl::KernelFunctor<cl::Buffer, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>;

        KernelType to_result(to_result_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        return std::pair<KernelType, cl::EnqueueArgs>(to_result_kernel, eargs);
//        heap_merge(eargs, workload, a_rows_pointers, a_cols, b_rows_compressed, b_rows_pointers, a_nzr, b_nzr);
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}



void matrix_multiplication(Controls &controls,
                           matrix_dcsr &matrix_out,
                           const matrix_coo &a,
                           const matrix_coo &b) {

    matrix_multiplication(controls, matrix_out, coo_to_dcsr_gpu(controls, a), coo_to_dcsr_gpu(controls, b)
    );
}


void matrix_multiplication(Controls &controls,
                           matrix_dcsr &matrix_out,
                           const matrix_dcsr &a,
                           const matrix_dcsr &b) {

    cl::Buffer nnz_estimation;
    count_workload(controls, nnz_estimation, a, b);
//    utils::print_gpu_buffer(controls, nnz_estimation, a.nzr());

    std::vector<cpu_buffer> cpu_workload_groups(BINS_NUM, cpu_buffer());
    cpu_buffer groups_pointers(BINS_NUM + 1);
    cpu_buffer groups_length(BINS_NUM);

    matrix_dcsr pre;
    build_groups_and_allocate_new_matrix(controls, pre, cpu_workload_groups, nnz_estimation, a, b.nCols());

    cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr());

    write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);
//    utils::print_gpu_buffer(controls, gpu_workload_groups,  a.nzr());

    run_kernels(controls, cpu_workload_groups, groups_length, groups_pointers,
                gpu_workload_groups, nnz_estimation,
                pre, a, b);

//    coo_utils::print_matrix(controls, pre);

    create_final_matrix(controls, matrix_out,
                        nnz_estimation, pre,
                        gpu_workload_groups, groups_pointers, groups_length,
                        a
                        );

//    utils::print_gpu_buffer(controls, matrix_out.rows_pointers_gpu(), matrix_out.nzr() + 1);
//    utils::print_gpu_buffer(controls, matrix_out.rows_compressed_gpu(), matrix_out.nzr());

}


void create_final_matrix(Controls &controls,
                         matrix_dcsr &c,
                         cl::Buffer &nnz_estimation,
                         const matrix_dcsr &pre,

                         const cl::Buffer &gpu_workload_groups,
                         const cpu_buffer &groups_pointers,
                         const cpu_buffer &groups_length,

                         const matrix_dcsr &a
                         ) {
    cl::Buffer c_rows_pointers;
    cl::Buffer c_rows_compressed;
    cl::Buffer c_cols_indices;

    uint32_t c_nnz;
    uint32_t c_nzr;

    /*
     * превращаем имеющийся массив nnz_estimation в корректные указатели типа csr массива.
     */
//    utils::print_gpu_buffer(controls, nnz_estimation, a.nzr());
    prefix_sum(controls, nnz_estimation, c_nnz, a.nzr());
//    utils::print_gpu_buffer(controls, nnz_estimation, a.nzr());
    c_cols_indices = cl::Buffer(controls.context, CL_TRUE, sizeof(uint32_t) * c_nnz);
    /*
     * заполняем послений элемент для работы как с указателями
     */
    controls.queue.enqueueWriteBuffer(nnz_estimation, CL_TRUE, sizeof(uint32_t) * a.nzr(), sizeof(uint32_t), &c_nnz);
//    utils::print_gpu_buffer(controls, nnz_estimation, a.nzr() + 1);

    if (groups_length[1] != 0) {
        auto single_value_rows_kernel = get_to_result_matrix_single_thread(controls, groups_length[1]);
        single_value_rows_kernel.first(single_value_rows_kernel.second,
                                       gpu_workload_groups, groups_pointers[1], groups_length[1],
                                       nnz_estimation, c_cols_indices, pre.rows_pointers_gpu(), pre.cols_indices_gpu());
    }

    uint32_t second_group_length = std::accumulate(groups_length.begin() + 2, groups_length.end(), 0u);

    if (second_group_length != 0) {
        auto ordinary_rows_kernel = get_to_result_matrix_work_group(controls, second_group_length);
        ordinary_rows_kernel.first(ordinary_rows_kernel.second,
                                   gpu_workload_groups, groups_pointers[2],
                                   nnz_estimation, c_cols_indices, pre.rows_pointers_gpu(), pre.cols_indices_gpu());
    }
//    utils::print_gpu_buffer(controls, c_cols_indices, c_nnz);
    /*
 * пока не испортили nnz_estimation, заберем информацию о нулевых рядах
 */
    cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr());
    // позиции -- 1 если элемент не 0
    prepare_positions(controls, positions, nnz_estimation, a.nzr(), "prepare_for_shift_empty_rows");
    // как нужно сдвинуть позиции, чтобы избавиться от информации о нулевых рядах.

    // ------------------------------------  get rid of empty rows -------------------------------
    prefix_sum(controls, positions, c_nzr, a.nzr());
    c_rows_pointers = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (c_nzr + 1));
    c_rows_compressed = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_nzr);
    set_positions(controls, c_rows_pointers, c_rows_compressed, nnz_estimation, a.rows_compressed_gpu(), positions,
                  c_nnz, a.nzr(), c_nzr);
//    utils::print_gpu_buffer(controls, c_rows_pointers, c_nzr + 1);
//    utils::print_gpu_buffer(controls, c_rows_compressed, c_nzr);
    c = matrix_dcsr(c_rows_pointers, c_rows_compressed, c_cols_indices, pre.nCols(), pre.nRows(), c_nnz, c_nzr);
}

void write_bins_info(Controls &controls,
                     cl::Buffer &gpu_workload_groups,
                     const std::vector<cpu_buffer> &cpu_workload_groups,
                     cpu_buffer &groups_pointers,
                     cpu_buffer &groups_length
                     ) {

    unsigned int offset = 0;
//    cl::Event end_write_buffer;
    for (uint32_t workload_group_id = 0; workload_group_id < BINS_NUM; ++workload_group_id) {
        const auto group = cpu_workload_groups[workload_group_id];
        if (group.empty()) continue;
        groups_pointers[workload_group_id] = offset;
        groups_length[workload_group_id] = group.size();
        controls.queue.enqueueWriteBuffer(gpu_workload_groups, CL_TRUE, sizeof(uint32_t) * offset, sizeof(uint32_t) * group.size(), group.data()
                                          /*, nullptr, &end_write_buffer*/);
        offset += group.size();
    }

    groups_pointers[BINS_NUM] = offset;
//    end_write_buffer.wait();
}

void run_kernels(Controls &controls,
                 const std::vector<cpu_buffer> &cpu_workload_groups,
                 const cpu_buffer &groups_length,
                 const cpu_buffer &groups_pointers,

                 const cl::Buffer &gpu_workload_groups,
                 cl::Buffer &nnz_estimation,

                 const matrix_dcsr &pre,
                 const matrix_dcsr &a,
                 const matrix_dcsr &b


) {
    for (uint32_t workload_group_id = 1; workload_group_id < BINS_NUM; ++workload_group_id) {
        const auto group = cpu_workload_groups[workload_group_id];
        if (group.empty()) continue;

        if (workload_group_id == 1) {
            auto kernelAndArgs = get_copy_one_value_kernel(controls, groups_length[workload_group_id]);
            kernelAndArgs.first(kernelAndArgs.second,
                                gpu_workload_groups, groups_pointers[workload_group_id], groups_length[workload_group_id],
                                pre.rows_pointers_gpu(), pre.cols_indices_gpu(),
                                a.rows_pointers_gpu(), a.cols_indices_gpu(),
                                b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                                b.nzr()
            );
            continue;
        }

        if (workload_group_id < 33 ) {
            auto kernelAndArgs = get_heap_kernel(controls, groups_length[workload_group_id],  workload_group_id);
            kernelAndArgs.first(kernelAndArgs.second,
                                gpu_workload_groups, groups_pointers[workload_group_id], groups_length[workload_group_id],
                                pre.rows_pointers_gpu(), pre.cols_indices_gpu(),
                                nnz_estimation,
                                a.rows_pointers_gpu(), a.cols_indices_gpu(),
                                b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                                b.nzr()
            );
            continue;
        }
    }
}

/*
 * cpu_workload_groups - indices of the rows, grouped in workload groups
 * cpu_workload - nnz estimation for each row of the result matrix
 * pre_rows_pointers -- указатели на начало соответствующего ряда новой матрицы.
 */

void build_groups_and_allocate_new_matrix(Controls& controls,
                                          matrix_dcsr &pre,
                                          std::vector<cpu_buffer>& cpu_workload_groups,
                                          cl::Buffer& nnz_estimation,
                                          const matrix_dcsr &a,
                                          uint32_t b_cols
                                          ) {
    cpu_buffer cpu_workload(a.nzr());
    controls.queue.enqueueReadBuffer(nnz_estimation, CL_TRUE, 0, sizeof(uint32_t) * a.nzr(), cpu_workload.data());

    uint32_t pre_nnz = 0;
    cpu_buffer rows_pointers_cpu(a.nzr() + 1);

    pre_nnz = 0;
    for (uint32_t i = 0; i < a.nzr(); ++i) {

        uint32_t current_workload = cpu_workload[i];
        uint32_t group = get_group(current_workload);
        cpu_workload_groups[group].push_back(i);

        rows_pointers_cpu[i] = pre_nnz;
        pre_nnz += current_workload < 513 ? current_workload : 256;
    }

    rows_pointers_cpu[a.nzr()] = pre_nnz;

    cl::Buffer pre_rows_pointers = cl::Buffer(controls.queue, rows_pointers_cpu.begin(), rows_pointers_cpu.end(), false);
    cl::Buffer pre_cols_indices_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * pre_nnz);

    pre = matrix_dcsr(pre_rows_pointers, a.rows_compressed_gpu(), pre_cols_indices_gpu,
                      a.nRows(), b_cols, pre_nnz, a.nzr());
}


/*
 * matches the nnz estimation of the row with one of 38 groups (indices from 0 to 37)
 */
uint32_t get_group(uint32_t size) {
    if (size < 33) return size;
    if (size < 65) return 33;
    if (size < 129) return 34;
    if (size < 257) return 35;
    if (size < 513) return 36;
    return 37;
}

uint32_t get_pre_size(uint32_t size) {
    if (size < 513) return size;
    // TODO : !parameter
    return 256;
}


void create_rows_pointers(Controls &controls,
                          cl::Buffer &rows_pointers_out,
                          cl::Buffer &rows_compressed_out,
                          const cl::Buffer &rows,
                          uint32_t size,
                          uint32_t &nzr // non zero rows
                          ) {

    cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
    prepare_positions(controls, positions, rows, size, "prepare_array_for_rows_positions");

//    utils::print_gpu_buffer(controls, positions, size);

    prefix_sum(controls, positions, nzr, size);
//    std::cout << "nzr: " << nzr << std::endl;
//    utils::print_gpu_buffer(controls, positions, size);

    cl::Buffer rows_pointers(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (nzr + 1));
    cl::Buffer rows_compressed(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * nzr);

    set_positions(controls, rows_pointers, rows_compressed, rows, positions, size, nzr);


    rows_pointers_out = std::move(rows_pointers);
    rows_compressed_out = std::move(rows_compressed);
}


void count_workload(Controls &controls,
                    cl::Buffer &nnz_estimation_out,
                    const matrix_dcsr &a,
                    const matrix_dcsr &b) {

    // буффер с распределением рабочей нагрузки, равен числу строк матрицы A
    cl::Program program;
    try {
        // a.nzr() + 1 to use it as pointers array later
        cl::Buffer nnz_estimation(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (a.nzr() + 1));
        program = controls.create_program_from_file("../src/coo/cl/count_workload.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());


        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, a.nzr());


        cl::Kernel coo_count_workload_kernel(program, "count_workload");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t> coo_count_workload(
                coo_count_workload_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        coo_count_workload(eargs, nnz_estimation, a.rows_pointers_gpu(), a.cols_indices_gpu(),
                           b.rows_compressed_gpu(), b.rows_pointers_gpu(), a.nzr(), b.nzr());

//        utils::print_gpu_buffer(controls, nnz_estimation, 10);
        nnz_estimation_out = std::move(nnz_estimation);

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}


void prepare_positions(Controls &controls,
                       cl::Buffer &positions,
                       const cl::Buffer &array,
                       uint32_t size,
                       const std::string &program_name
) {
    cl::Program program;
    try {
        program = controls.create_program_from_file("../src/coo/cl/prepare_positions.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

//        std::vector<uint32_t> look_positions(merged_size);

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, size);


//        cl::Kernel coo_prepare_positions_kernel(program, "prepare_array_for_rows_positions");
        cl::Kernel coo_prepare_positions_kernel(program, program_name.c_str());
        cl::KernelFunctor<cl::Buffer, cl::Buffer, uint32_t> coo_prepare_positions(
                coo_prepare_positions_kernel);
        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        coo_prepare_positions(eargs, positions, array, size);

//        controls.queue.enqueueReadBuffer(positions, CL_TRUE, 0, sizeof(uint32_t) * merged_size, look_positions.data());

        std::cout << "\nprepare positions finished\n";

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}

void set_positions(Controls &controls,
                   cl::Buffer &rows_pointers,
                   cl::Buffer &rows_compressed,
                   const cl::Buffer &rows,
                   const cl::Buffer &positions,
                   uint32_t size,
                   uint32_t nzr
                   ) {

    cl::Program program;
    try {
        program = controls.create_program_from_file("../src/coo/cl/set_positions.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        cl::Kernel set_positions_kernel(program, "set_positions_rows");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, unsigned int, unsigned int> set_positions(
                set_positions_kernel);

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, size);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        set_positions(eargs, rows_pointers, rows_compressed, rows, positions, size, nzr);
//        utils::print_gpu_buffer(controls, _rows_compressed, std::min(nzr, 10U));
        std::cout << "\nset_positions finished\n";

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}


void set_positions(Controls &controls,
                   cl::Buffer &c_rows_pointers,
                   cl::Buffer &c_rows_compressed,
                   const cl::Buffer &nnz_estimation,
                   const cl::Buffer &a_rows_compressed,
                   const cl::Buffer &positions,
                   uint32_t c_nnz,
                   uint32_t old_nzr,
                   uint32_t c_nzr
) {

    cl::Program program;
    try {
        program = controls.create_program_from_file("../src/coo/cl/set_positions.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        cl::Kernel set_positions_kernel(program, "set_positions_pointers_and_rows");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                            unsigned int, unsigned int, unsigned int> set_positions(
                set_positions_kernel);

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, old_nzr);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        set_positions(eargs, c_rows_pointers, c_rows_compressed, nnz_estimation, a_rows_compressed, positions, c_nnz, old_nzr, c_nzr  );
//        utils::print_gpu_buffer(controls, _rows_compressed, std::min(nzr, 10U));
        std::cout << "\nset_positions finished\n";

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}