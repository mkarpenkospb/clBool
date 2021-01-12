#include "coo_matrix_multiplication.hpp"
#include "coo_matrix_addition.hpp"
#include "../library_classes/controls.hpp"
#include "../utils.hpp"
#include "../library_classes/matrix_coo.hpp"

const uint32_t BINS_NUM = 38;
typedef std::vector<uint32_t> cpu_buffer;




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

    // Распределять кернелы будем на CPU
    cpu_buffer cpu_workload(a_nzr);
    controls.queue.enqueueReadBuffer(workload, CL_TRUE, 0, sizeof(uint32_t) * a_nzr, cpu_workload.data());

    std::vector<cpu_buffer> cpu_workload_groups(a_nzr, cpu_buffer());
    cpu_buffer groups_pointers(BINS_NUM);
    build_groups(cpu_workload_groups, cpu_workload);
    cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_nzr);
    unsigned int offset = 0;


    for (const auto &group: cpu_workload_groups) {
        controls.queue.enqueueWriteBuffer(gpu_workload_groups, CL_TRUE, offset, group.size(), group.data());
        offset += group.size();
    }

}


/*
 * workload_groups - indices of the rows, grouped in workload groups
 * cpu_workload - nnz estimation for each row of the result matrix
 */
void build_groups(std::vector<cpu_buffer>& workload_groups, const cpu_buffer& cpu_workload) {
    const auto size = cpu_workload.size();
    for (uint32_t i = 0; i < size; ++i) {
        uint32_t group = get_group(cpu_workload[i]);
        workload_groups[group].push_back(i);
    }
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


void create_rows_pointers(Controls &controls,
                          cl::Buffer &rows_pointers_out,
                          cl::Buffer &rows_compressed_out,
                          const cl::Buffer &rows,
                          uint32_t size,
                          uint32_t &nzr // non zero rows
                          ) {

    cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
    prepare_positions(controls, positions, rows, size);

    utils::print_gpu_buffer(controls, positions, size);

    prefix_sum(controls, positions, nzr, size);

    utils::print_gpu_buffer(controls, positions, size);

    cl::Buffer rows_pointers(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (nzr + 1));
    cl::Buffer rows_compressed(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * nzr);

    set_positions(controls, rows_pointers, rows_compressed, rows, positions, size, nzr);


    rows_pointers_out = std::move(rows_pointers);
    rows_compressed_out = std::move(rows_compressed);
}


void count_workload(Controls &controls,
                    cl::Buffer workload_out,
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