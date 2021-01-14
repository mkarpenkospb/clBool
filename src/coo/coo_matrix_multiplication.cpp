#include "coo_matrix_multiplication.hpp"
#include "coo_matrix_addition.hpp"
#include "../library_classes/controls.hpp"
#include "../utils.hpp"
#include "../library_classes/matrix_coo.hpp"

const uint32_t BINS_NUM = 38;
const uint32_t HEAP_MERGE_BLOCK_SIZE = 32;
typedef std::vector<uint32_t> cpu_buffer;

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
        uint32_t block_size = std::min(controls.block_size, utils::ceil_to_power2(group_length));

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

//        program = controls.create_program_from_file("../src/coo/cl/prepare_positions.cl");
//        uint32_t block_size = controls.block_size;
//
//        std::stringstream options;
//        options << "-D GROUP_SIZE=" << block_size;
//        program.build(options.str().c_str());


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


void matrix_multiplication(Controls &controls,
                           matrix_coo &matrix_out,
                           const matrix_coo &a,
                           const matrix_coo &b) {

    cl::Buffer a_rows_pointers;
    cl::Buffer b_rows_pointers;

    /*
     * rows_compressed -- rows array with no duplicates
     * probably we don't need it
     */
    cl::Buffer a_rows_compressed;
    cl::Buffer b_rows_compressed;

    uint32_t a_nzr;
    uint32_t b_nzr;

    create_rows_pointers(controls, a_rows_pointers, a_rows_compressed, a.rows_indices_gpu(), a.nnz(), a_nzr);
    create_rows_pointers(controls, b_rows_pointers, b_rows_compressed, b.rows_indices_gpu(), b.nnz(), b_nzr);

    cl::Buffer workload;

    count_workload(controls, workload,
                   a_rows_pointers, a.cols_indices_gpu(),
                   b_rows_compressed, b_rows_pointers, b.cols_indices_gpu(),
                   a_nzr, b_nzr);

 // ----------------------------------------------- точка тестирования ----------------------------------------

    cpu_buffer cpu_workload(a_nzr);
    controls.queue.enqueueReadBuffer(workload, CL_TRUE, 0, sizeof(uint32_t) * a_nzr, cpu_workload.data());

    std::vector<cpu_buffer> cpu_workload_groups(BINS_NUM, cpu_buffer());
    cpu_buffer groups_pointers(BINS_NUM + 1);
    cpu_buffer groups_length(BINS_NUM);

    /*
     * Промежуточная матрица будет CSR относительно обнулившихся рядов,
     */

    /*
     * Создадим новую матрицу pre - промежуточну.
     * Тут небольшая каша, функция, формирующая группы смешивается с функцией для
     * аллокации новой матрицы.
     */

    uint32_t pre_nnz;
    cl::Buffer pre_rows_pointers;
    cl::Buffer pre_cols_indices_gpu;
    build_groups_and_allocate_new_matrix(controls,
                                         pre_rows_pointers, pre_cols_indices_gpu, pre_nnz,
                                         cpu_workload_groups, cpu_workload, a_nzr);

    /*
     * Прямо тут можно независимо вызывать кернелы. Есди можно конечно одновременн писать и читать из буфера.
     * !! Может, нужно сделать несколько буферов. 38 штук и не париться с ними
     * Но можно ли вызвать их независимо?
     */

    cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_nzr);

    write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);

    run_kernels(controls, cpu_workload_groups, groups_length, groups_pointers,
                gpu_workload_groups, workload,
                pre_rows_pointers, pre_cols_indices_gpu,
                a_rows_pointers, a.cols_indices_gpu(),
                b_rows_pointers, b_rows_compressed, b.cols_indices_gpu(),
                b_nzr
                );

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

                 const cl::Buffer &pre_rows_pointers,
                 cl::Buffer &pre_cols_indices_gpu,

                 const cl::Buffer &a_rows_pointers,
                 const cl::Buffer &a_cols,

                 const cl::Buffer &b_rows_pointers,
                 const cl::Buffer &b_rows_compressed,
                 const cl::Buffer &b_cols,

                 uint32_t b_nzr
) {
    for (uint32_t workload_group_id = 1; workload_group_id < BINS_NUM; ++workload_group_id) {
        const auto group = cpu_workload_groups[workload_group_id];
        if (group.empty()) continue;

        if (workload_group_id == 1) {
            auto kernelAndArgs = get_copy_one_value_kernel(controls, groups_length[workload_group_id]);
            kernelAndArgs.first(kernelAndArgs.second,
                                gpu_workload_groups, groups_pointers[workload_group_id], groups_length[workload_group_id],
                                pre_rows_pointers, pre_cols_indices_gpu,
                                a_rows_pointers, a_cols,
                                b_rows_pointers, b_rows_compressed, b_cols,
                                b_nzr
            );
            continue;
        }

        if (workload_group_id < 33 ) {
            auto kernelAndArgs = get_heap_kernel(controls, groups_length[workload_group_id],  workload_group_id);
            kernelAndArgs.first(kernelAndArgs.second,
                                gpu_workload_groups, groups_pointers[workload_group_id], groups_length[workload_group_id],
                                pre_rows_pointers, pre_cols_indices_gpu,
                                nnz_estimation,
                                a_rows_pointers, a_cols,
                                b_rows_pointers, b_rows_compressed, b_cols,
                                b_nzr
            );
            continue;
        }
    }
}

/*
 * workload_groups - indices of the rows, grouped in workload groups
 * cpu_workload - nnz estimation for each row of the result matrix
 * pre_rows_pointers -- указатели на начало соответствующего ряда новой матрицы.
 */

void build_groups_and_allocate_new_matrix(Controls& controls,
                                          cl::Buffer& pre_rows_pointers,
                                          cl::Buffer& pre_cols_indices_gpu,
                                          uint32_t &pre_nnz,
                                          std::vector<cpu_buffer>& workload_groups,
                                          const cpu_buffer& cpu_workload,
                                          uint32_t a_nzr
                                          ) {
    cpu_buffer rows_pointers_cpu(a_nzr + 1);
    pre_nnz = 0;
    for (uint32_t i = 0; i < a_nzr; ++i) {

        uint32_t current_workload = cpu_workload[i];
        uint32_t group = get_group(current_workload);
        workload_groups[group].push_back(i);

        rows_pointers_cpu[i] = pre_nnz;
        pre_nnz += current_workload < 513 ? current_workload : 256;
    }

    rows_pointers_cpu[a_nzr] = pre_nnz;

    pre_rows_pointers = cl::Buffer(controls.queue, rows_pointers_cpu.begin(), rows_pointers_cpu.end(), false);
    pre_cols_indices_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * pre_nnz);
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
    prepare_positions(controls, positions, rows, size);

//    utils::print_gpu_buffer(controls, positions, size);

    prefix_sum(controls, positions, nzr, size);

//    utils::print_gpu_buffer(controls, positions, size);

    cl::Buffer rows_pointers(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (nzr + 1));
    cl::Buffer rows_compressed(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * nzr);

    set_positions(controls, rows_pointers, rows_compressed, rows, positions, size, nzr);


    rows_pointers_out = std::move(rows_pointers);
    rows_compressed_out = std::move(rows_compressed);
}


void count_workload(Controls &controls,
                    cl::Buffer &workload_out,
                    cl::Buffer &a_rows_pointers,
                    const cl::Buffer &a_cols,
                    cl::Buffer &b_rows_compressed,
                    cl::Buffer &b_rows_pointers,
                    const cl::Buffer &b_cols,
                    uint32_t a_nzr,
                    uint32_t b_nzr) {

    // буффер с распределением рабочей нагрузки, равен числу строк матрицы A
    cl::Program program;
    try {
        cl::Buffer workload(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_nzr);
        program = controls.create_program_from_file("../src/coo/cl/count_workload.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());


        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, a_nzr);


        cl::Kernel coo_count_workload_kernel(program, "count_workload");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t> coo_count_workload(
                coo_count_workload_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        coo_count_workload(eargs, workload, a_rows_pointers, a_cols, b_rows_compressed, b_rows_pointers, a_nzr, b_nzr);

//        utils::print_gpu_buffer(controls, workload, 10);
        workload_out = std::move(workload);

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
                       const cl::Buffer &rows,
                       uint32_t size
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


        cl::Kernel coo_prepare_positions_kernel(program, "prepare_array_for_rows_positions");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, uint32_t> coo_prepare_positions(
                coo_prepare_positions_kernel);
        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        coo_prepare_positions(eargs, positions, rows, size);

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
//        utils::print_gpu_buffer(controls, rows_compressed, std::min(nzr, 10U));
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