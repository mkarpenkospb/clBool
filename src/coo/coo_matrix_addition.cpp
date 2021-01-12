
#include "../library_classes/controls.hpp"
#include "../utils.hpp"
#include "coo_utils.hpp"
#include "coo_matrix_addition.hpp"


void matrix_addition(Controls &controls,
                     matrix_coo &matrix_out,
                     const matrix_coo &a,
                     const matrix_coo &b) {

    cl::Buffer merged_rows;
    cl::Buffer merged_cols;
    size_t new_size;

    merge(controls, merged_rows, merged_cols, a, b);

    reduce_duplicates(controls, merged_rows, merged_cols, new_size, a.nnz() + b.nnz());

    matrix_out = matrix_coo(controls, std::max(a.nRows(), b.nRows()), std::max(a.nCols(), b.nCols()), new_size,
                            std::move(merged_rows), std::move(merged_cols));

}


void merge(Controls &controls,
           cl::Buffer &merged_rows_out,
           cl::Buffer &merged_cols_out,
           const matrix_coo &a,
           const matrix_coo &b) {

    cl::Program program;

    try {

        uint32_t merged_size = a.nnz() + b.nnz();

        program = controls.create_program_from_file("../src/coo/cl/merge_path.cl");

        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, merged_size);

        cl::Buffer merged_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);
        cl::Buffer merged_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

        cl::Kernel coo_merge(program, "merge");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t, uint32_t> coo_merge_kernel(coo_merge);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        coo_merge_kernel(eargs,
                         merged_rows, merged_cols,
                         a.rows_indices_gpu(), a.cols_indices_gpu(),
                         b.rows_indices_gpu(), b.cols_indices_gpu(),
                         a.nnz(), b.nnz());

        // TODO: maybe add wait
        check_merge_correctness(controls, merged_rows, merged_cols, merged_size);

        merged_rows_out = std::move(merged_rows);
        merged_cols_out = std::move(merged_cols);
        std::cout << "\nmerge finished\n";
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}


void reduce_duplicates(Controls &controls,
                       cl::Buffer &merged_rows,
                       cl::Buffer &merged_cols,
                       uint32_t &new_size,
                       uint32_t merged_size
) {
    // ------------------------------------ prepare array to count positions ----------------------

    cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

    prepare_positions(controls, positions, merged_rows, merged_cols, merged_size);

    // ------------------------------------ calculate positions, get new_size -----------------------------------

    prefix_sum(controls, positions, new_size, merged_size);

    cl::Buffer new_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_size);
    cl::Buffer new_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_size);

    set_positions(controls, new_rows, new_cols, merged_rows, merged_cols, positions, merged_size);

    merged_rows = std::move(new_rows);
    merged_cols = std::move(new_cols);

    std::cout << "\nreduce finished\n";
}


void prepare_positions(Controls &controls,
                       cl::Buffer &positions,
                       cl::Buffer &merged_rows,
                       cl::Buffer &merged_cols,
                       uint32_t merged_size
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
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, merged_size);

        cl::Kernel coo_prepare_positions_kernel(program, "prepare_array_for_positions");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t> coo_prepare_positions(
                coo_prepare_positions_kernel);
        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        coo_prepare_positions(eargs, positions, merged_rows, merged_cols, merged_size);

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

void prefix_sum(Controls &controls,
                cl::Buffer &positions,
                uint32_t &new_size,
                uint32_t merged_size) {
    cl::Program program;
    try {
        program = controls.create_program_from_file("../src/coo/cl/prefix_sum.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, merged_size);

        uint32_t a_size = (merged_size + block_size - 1) / block_size; // max to save first roots
        uint32_t b_size = (a_size + block_size - 1) / block_size; // max to save second roots

        cl::Buffer a_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a_size);
        cl::Buffer b_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * b_size);
        cl::LocalSpaceArg local_array = cl::Local(sizeof(uint32_t) * block_size);

        // prefix sum step kernel
        cl::Kernel scan_kernel(program, "scan_blelloch");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, unsigned int> scan(scan_kernel);

        cl::Kernel update_kernel(program, "update_pref_sum");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, unsigned int, unsigned int> update(update_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        // for test correctness
        std::vector<uint32_t> before(merged_size, 0);
        std::vector<uint32_t> result(merged_size, 0);

        controls.queue.enqueueReadBuffer(positions, CL_TRUE, 0, sizeof(uint32_t) * merged_size, before.data());
        // zero step to count blockSize prefixes on future result array

        uint32_t leaf_size = 1;
        cl::Event event = scan(eargs, a_gpu, positions, local_array, merged_size);
        event.wait();

        uint32_t outer = (merged_size + block_size - 1) / block_size;

        cl::Buffer *a_gpu_ptr = &a_gpu;
        cl::Buffer *b_gpu_ptr = &b_gpu;

        unsigned int *a_size_ptr = &a_size;
        unsigned int *b_size_ptr = &b_size;

        while (outer > 1) {
            leaf_size *= block_size;
            cl::EnqueueArgs eargs_in_recursion(controls.queue,
                                               cl::NDRange((outer + work_group_size - 1) / work_group_size *
                                                           work_group_size),
                                               cl::NDRange(work_group_size));

            cl::Event event_in_recursion = scan(eargs_in_recursion, *b_gpu_ptr, *a_gpu_ptr, local_array, outer);
            event_in_recursion.wait();

            cl::Event update_event = update(eargs, positions, *a_gpu_ptr, merged_size, leaf_size);
            update_event.wait();

            outer = (outer + block_size - 1) / block_size;
            std::swap(a_gpu_ptr, b_gpu_ptr);
            std::swap(a_size_ptr, b_size_ptr);
        }

//        controls.queue.enqueueReadBuffer(positions, CL_TRUE, 0, sizeof(uint32_t) * merged_size, result.data());

        // the last element of positions is the new matrix size - 1
        controls.queue.enqueueReadBuffer(positions, CL_TRUE, (merged_size - 1) * sizeof(uint32_t),
                                         sizeof(uint32_t), &new_size);
        new_size++;
//        check_pref_correctness(result, before);
        std::cout << "\nprefix sum finished\n";
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
                   cl::Buffer &new_rows,
                   cl::Buffer &new_cols,
                   cl::Buffer &merged_rows,
                   cl::Buffer &merged_cols,
                   cl::Buffer &positions,
                   uint32_t merged_size) {

    cl::Program program;
    try {
        program = controls.create_program_from_file("../src/coo/cl/set_positions.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        cl::Kernel set_positions_kernel(program, "set_positions");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, unsigned int> set_positions(
                set_positions_kernel);

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, merged_size);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        set_positions(eargs, new_rows, new_cols, merged_rows, merged_cols, positions, merged_size);

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


void check_pref_correctness(const std::vector<uint32_t> &result,
                            const std::vector<uint32_t> &before) {
    uint32_t n = before.size();
    uint32_t acc = 0;

    for (uint32_t i = 0; i < n; ++i) {
        acc = i == 0 ? before[i] : before[i] + acc;

        if (acc != result[i]) {
            throw std::runtime_error("incorrect result");
        }
    }
    std::cout << "correct pref sum, the last value is " << result[n - 1] << std::endl;
}


// check weak correctness
void check_merge_correctness(Controls &controls, cl::Buffer &rows, cl::Buffer &cols, size_t merged_size) {
    std::vector<uint32_t> rowsC(merged_size);
    std::vector<uint32_t> colsC(merged_size);

    controls.queue.enqueueReadBuffer(rows, CL_TRUE, 0, sizeof(uint32_t) * merged_size, rowsC.data());
    controls.queue.enqueueReadBuffer(cols, CL_TRUE, 0, sizeof(uint32_t) * merged_size, colsC.data());

    coo_utils::check_correctness(rowsC, colsC);
}
