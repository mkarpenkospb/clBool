
#include "../library_classes/controls.hpp"
#include "../library_classes/matrix_coo.hpp"
#include "../utils.hpp"
#include "coo_utils.hpp"
#include "coo_matrix_addition.hpp"

void check_merge_correctness(Controls& controls, const matrix_coo& c);

void addition(Controls& controls, matrix_coo& c, const matrix_coo& a, const matrix_coo& b) {
    cl::Program program;

    try {
        // ---------------------------------------- merge ---------------------------------------

        program = controls.create_program_from_file("../src/COO/cl/merge_path.cl");

        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());

        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, a.get_n_entities() + b.get_n_entities());


        cl::Kernel coo_merge(program, "merge");
        cl::KernelFunctor</*c: */ cl::Buffer, cl::Buffer, /*a: */ cl::Buffer, cl::Buffer, /*b: */ cl::Buffer, cl::Buffer,
                          uint32_t, uint32_t> coo_merge_kernel(coo_merge);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        coo_merge_kernel(eargs,
                  c.get_rows_indexes_gpu(), c.get_cols_indexes_gpu(),
                  a.get_rows_indexes_gpu(), a.get_cols_indexes_gpu(),
                  b.get_rows_indexes_gpu(), b.get_cols_indexes_gpu(),
                  a.get_n_entities(), b.get_n_entities());

        // TODO: maybe add wait
        check_merge_correctness(controls, c);

    } catch (const cl::Error& e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}

// check weak correctness
void check_merge_correctness(Controls& controls, const matrix_coo& c) {
    std::vector<uint32_t> rowsC(c.get_n_entities());
    std::vector<uint32_t> colsC(c.get_n_entities());

    controls.queue.enqueueReadBuffer(c.get_rows_indexes_gpu(), CL_TRUE, 0, sizeof(uint32_t) * c.get_n_entities(), rowsC.data());
    controls.queue.enqueueReadBuffer(c.get_cols_indexes_gpu(), CL_TRUE, 0, sizeof(uint32_t) * c.get_n_entities(), colsC.data());

    coo_utils::check_correctness(rowsC, colsC);
}


