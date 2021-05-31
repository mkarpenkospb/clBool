#include <numeric>
#include <sstream>
#include "dcsr_matrix_multiplication.hpp"

namespace clbool::dcsr {
    const uint32_t BINS_NUM = 38;
    const uint32_t HEAP_MERGE_BLOCK_SIZE = 32;

    uint32_t esc_estimation(uint32_t group) {
        switch (group) {
            case 33:
                return 64;
            case 34:
                return 128;
            case 35:
                return 256;
            case 36:
                return 512;
            default:
                std::stringstream s;
                s << "Invalid group: " << group << ". Possible values from 33 to 36 inc.";
                throw Exception(s.str(), CLBOOL_INVALID_ARGUMENT, 347113, __FILE__, __FUNCTION__, __LINE__);
        }
    }

    void matrix_multiplication(Controls &controls,
                               matrix_dcsr &matrix_out,
                               const matrix_dcsr &a,
                               const matrix_dcsr &b) {

        if (a.ncols() != b.nrows()) {
            std::stringstream s;
            s << "Invalid input matrix size! a : " << a.nrows() << " x " << a.ncols()
            << ", b: " << b.nrows() << " x " << b.ncols();
            throw Exception(s.str(), CLBOOL_INVALID_ARGUMENT, 361621, __FILE__, __FUNCTION__, __LINE__);
        }

        if (a.nnz() == 0 || b.nnz() == 0) {
            matrix_out = matrix_dcsr(a.nrows(), b.ncols());
            return;
        }
        cl::Buffer nnz_estimation;
        count_workload(controls, nnz_estimation, a, b);

        std::vector<cpu_buffer> cpu_workload_groups(BINS_NUM, cpu_buffer());
        cpu_buffer groups_pointers(BINS_NUM + 1);
        cpu_buffer groups_length(BINS_NUM);

        cl::Buffer aux_37_group_mem_pointers;
        cl::Buffer aux_37_group_mem;

        matrix_dcsr pre;
        build_groups_and_allocate_new_matrix(controls, pre, cpu_workload_groups, nnz_estimation, a, b.ncols(),
                                             aux_37_group_mem_pointers, aux_37_group_mem);
        if (pre.empty()) {
            matrix_out = matrix_dcsr(a.nrows(), b.ncols());
            return;
        }

        cl::Buffer gpu_workload_groups;
        CHECK_RUN(gpu_workload_groups = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr()),
                  684621);

        write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);


        run_kernels(controls, groups_length, groups_pointers,
                    gpu_workload_groups, nnz_estimation,
                    pre, a, b,
                    aux_37_group_mem_pointers, aux_37_group_mem
        );


        create_final_matrix(controls, matrix_out,
                            nnz_estimation, pre,
                            gpu_workload_groups, groups_pointers, groups_length,
                            a
        );
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
        cl::Buffer c_rpt;
        cl::Buffer c_rows;
        cl::Buffer c_cols_indices;

        uint32_t c_nnz;
        uint32_t c_nzr;

        prefix_sum(controls, nnz_estimation, c_nnz, a.nzr() + 1);
        CHECK_CL(c_cols_indices = cl::Buffer(controls.context, CL_TRUE, sizeof(uint32_t) * c_nnz),
                 CLBOOL_CREATE_BUFFER_ERROR, 2808182);

        cl::Event e1;
        cl::Event e2;
        if (groups_length[1] != 0) {
            auto single_value_rows = kernel<cl::Buffer, uint32_t, uint32_t,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                    ("to_result_matrix_single_thread", "to_result");
            single_value_rows.set_block_size(
                    std::min(controls.block_size, std::max(32u, utils::ceil_to_power2(groups_length[1]))));
            single_value_rows.set_work_size(groups_length[1]);
            single_value_rows.set_async(true);
            CHECK_RUN(e1 = single_value_rows.run(controls, gpu_workload_groups, groups_pointers[1], groups_length[1],
                                                 nnz_estimation, c_cols_indices, pre.rpt_gpu(), pre.cols_gpu()),
                      10923842);
        }

        uint32_t second_group_length = std::accumulate(groups_length.begin() + 2, groups_length.end(), 0u);

        if (second_group_length != 0) {
            auto ordinary_rows = kernel<cl::Buffer, uint32_t,
                    cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
                    ("to_result_matrix_work_group", "to_result");
            ordinary_rows.set_work_size(controls.block_size * second_group_length);
            ordinary_rows.set_async(true);
            CHECK_RUN(e2 = ordinary_rows.run(controls,
                                             gpu_workload_groups, groups_length[0] + groups_length[1],
                                             nnz_estimation, c_cols_indices, pre.rpt_gpu(), pre.cols_gpu()), 539423431);
        }


        if (groups_length[1] != 0) {
            CHECK_CL(e1.wait(), CLBOOL_EVENT_WAITING_ERROR, 318681);
        }
        if (second_group_length != 0) {
            CHECK_CL(e2.wait(), CLBOOL_EVENT_WAITING_ERROR, 235681);
        }


        cl::Buffer positions;
        CHECK_CL(positions = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (a.nzr() + 1)),
                 CLBOOL_CREATE_BUFFER_ERROR, 1029421)

        prepare_positions(controls, positions, nnz_estimation, a.nzr(), "prepare_for_shift_empty_rows");

        // ------------------------------------  get rid of empty rows_gpu -------------------------------

        prefix_sum(controls, positions, c_nzr, a.nzr() + 1);
        CHECK_CL(c_rpt = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (c_nzr + 1)),
                 CLBOOL_CREATE_BUFFER_ERROR, 1868261);
        CHECK_CL(c_rows = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_nzr),
                 CLBOOL_CREATE_BUFFER_ERROR, 1243111);
        set_positions(controls, c_rpt, c_rows, nnz_estimation, a.rows_gpu(), positions, a.nzr());

        c = matrix_dcsr(c_rpt, c_rows, c_cols_indices, pre.nrows(), pre.ncols(), c_nnz, c_nzr);
    }

    void write_bins_info(Controls &controls,
                         cl::Buffer &gpu_workload_groups,
                         const std::vector<cpu_buffer> &cpu_workload_groups,
                         cpu_buffer &groups_pointers,
                         cpu_buffer &groups_length
    ) {

        uint32_t offset = 0;
        uint32_t bins = cpu_workload_groups.size();
        for (uint32_t workload_group_id = 0; workload_group_id < bins; ++workload_group_id) {
            const cpu_buffer &group = cpu_workload_groups[workload_group_id];
            if (group.empty()) continue;
            groups_pointers[workload_group_id] = offset;
            groups_length[workload_group_id] = group.size();
            CHECK_CL(controls.queue.enqueueWriteBuffer(gpu_workload_groups, CL_TRUE, sizeof(uint32_t) * offset,
                                                       sizeof(uint32_t) * group.size(), group.data()),
                     CLBOOL_WRITE_BUFFER_ERROR, 19374532);

            offset += group.size();
        }

        groups_pointers[bins] = offset;
    }

    void run_kernels(Controls &controls,
                     const cpu_buffer &groups_length,
                     const cpu_buffer &groups_pointers,

                     const cl::Buffer &gpu_workload_groups,
                     cl::Buffer &nnz_estimation,

                     const matrix_dcsr &pre,
                     const matrix_dcsr &a,
                     const matrix_dcsr &b,

                     const cl::Buffer &aux_mem_pointers,
                     cl::Buffer &aux_mem

    ) {
        auto heap_merge = kernel<cl::Buffer, uint32_t, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t>("heap_merge", "heap_merge");
        heap_merge.set_block_size(HEAP_MERGE_BLOCK_SIZE);
        heap_merge.set_async(true);

        auto copy_one_value = kernel<cl::Buffer, uint32_t, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t>("copy_one_value", "copy_one_value");
        copy_one_value.set_kernel_name("copy_one_value");
        copy_one_value.set_async(true);

        auto merge_large_rows = kernel<cl::Buffer, uint32_t, cl::Buffer,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t>("merge_large_rows", "merge_large_rows");
        merge_large_rows.set_block_size(controls.block_size);
        merge_large_rows.set_async(true);

        auto esc_kernel = kernel<cl::Buffer, uint32_t, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t>("bitonic_esc", "bitonic_esc");
        esc_kernel.set_block_size(controls.block_size);
        esc_kernel.set_async(true);


        std::vector<cl::Event> events;
        for (uint32_t workload_group_id = 1; workload_group_id < BINS_NUM; ++workload_group_id) {
            if (groups_length[workload_group_id] == 0) continue;

            if (workload_group_id == 1) {
                LOG << "first group";
                copy_one_value.set_work_size(groups_length[workload_group_id])
                        .set_block_size(std::min(controls.block_size,
                                                 std::max(32u,
                                                          utils::ceil_to_power2(groups_length[workload_group_id]))));

                cl::Event ev;
                CHECK_RUN(ev = copy_one_value.run(controls,
                                                  gpu_workload_groups, groups_pointers[workload_group_id],
                                                  groups_length[workload_group_id],
                                                  pre.rpt_gpu(), pre.cols_gpu(),
                                                  a.rpt_gpu(), a.cols_gpu(),
                                                  b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
                                                  b.nzr()), 9281892);

                events.push_back(ev);
                continue;
            }


            if (workload_group_id < 33) {
                LOG << "2 - 32: " << workload_group_id;
                heap_merge.set_work_size(groups_length[workload_group_id])
                        .add_option("NNZ_ESTIMATION", workload_group_id);

                cl::Event ev;
                CHECK_RUN(ev = heap_merge.run(controls, gpu_workload_groups, groups_pointers[workload_group_id],
                                              groups_length[workload_group_id],
                                              pre.rpt_gpu(), pre.cols_gpu(),
                                              nnz_estimation,
                                              a.rpt_gpu(), a.cols_gpu(),
                                              b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
                                              b.nzr()), 8048904);
                events.push_back(ev);

                continue;
            }

            if (workload_group_id < 37) {
                LOG << "33 - 36";
                uint32_t block_size = std::max(32u, esc_estimation(workload_group_id) / 2);
                esc_kernel.add_option("NNZ_ESTIMATION", esc_estimation(workload_group_id))
                        .set_block_size(block_size)
                        .set_work_size(block_size * groups_length[workload_group_id]);

                cl::Event ev;
                CHECK_RUN(ev = esc_kernel.run(
                        controls,
                        gpu_workload_groups, groups_pointers[workload_group_id],
                        groups_length[workload_group_id],
                        pre.rpt_gpu(), pre.cols_gpu(),
                        nnz_estimation,
                        a.rpt_gpu(), a.cols_gpu(),
                        b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
                        b.nzr()), 143125391)
                events.push_back(ev);
                continue;
            }


            LOG << "37!";
            merge_large_rows.set_work_size(groups_length[workload_group_id] * controls.block_size);
            cl::Event ev;
            CHECK_RUN(ev = merge_large_rows.run(controls,
                                                gpu_workload_groups, groups_pointers[workload_group_id],
                                                aux_mem_pointers, aux_mem,
                                                pre.rpt_gpu(), pre.cols_gpu(),
                                                nnz_estimation,
                                                a.rpt_gpu(), a.cols_gpu(),
                                                b.rpt_gpu(), b.rows_gpu(), b.cols_gpu(),
                                                b.nzr()), 648361)
            events.push_back(ev);
        }

        CHECK_CL(cl::Event::waitForEvents(events), CLBOOL_EVENT_WAITING_ERROR, 121343321);
    }

    void build_groups_and_allocate_new_matrix(Controls &controls,
                                              matrix_dcsr &pre,
                                              std::vector<cpu_buffer> &cpu_workload_groups,
                                              cl::Buffer &nnz_estimation,
                                              const matrix_dcsr &a,
                                              uint32_t b_cols,

                                              cl::Buffer &aux_pointers,
                                              cl::Buffer &aux_mem
    ) {

        cpu_buffer aux_pointers_cpu;
        uint32_t aux = 0;

        cpu_buffer cpu_workload(a.nzr());

        CHECK_CL(controls.queue.enqueueReadBuffer(nnz_estimation, CL_TRUE, 0, sizeof(uint32_t) * a.nzr(),
                                                 cpu_workload.data()),
                CLBOOL_READ_BUFFER_ERROR, 32453245)


        uint32_t pre_nnz = 0;
        cpu_buffer rows_pointers_cpu(a.nzr() + 1);

        pre_nnz = 0;
        for (uint32_t i = 0; i < a.nzr(); ++i) {

            uint32_t current_workload = cpu_workload[i];
            uint32_t group = get_group(current_workload);
            cpu_workload_groups[group].push_back(i);
            rows_pointers_cpu[i] = pre_nnz;

            pre_nnz += current_workload;
            if (group == 37) {
                aux_pointers_cpu.push_back(aux);
                aux += current_workload;
            }
        }
        if (pre_nnz == 0) {
            pre = matrix_dcsr(a.nrows(), b_cols);
            return;
        }
        aux_pointers_cpu.push_back(aux);
        rows_pointers_cpu[a.nzr()] = pre_nnz;

        cl::Buffer pre_rows_pointers;
        CHECK_CL(pre_rows_pointers = cl::Buffer(controls.queue, rows_pointers_cpu.begin(), rows_pointers_cpu.end(),
                                                false),
                 CLBOOL_CREATE_BUFFER_ERROR, 46292371);

        cl::Buffer pre_cols_indices_gpu;
        CHECK_CL(pre_cols_indices_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * pre_nnz),
                 CLBOOL_CREATE_BUFFER_ERROR, 36571649);

        if (aux != 0) {
            CHECK_CL(aux_pointers = cl::Buffer(controls.queue, aux_pointers_cpu.begin(), aux_pointers_cpu.end(), true),
                     CLBOOL_CREATE_BUFFER_ERROR, 830183094)

            CHECK_CL(aux_mem = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * aux),
                     CLBOOL_CREATE_BUFFER_ERROR, 63912211);
        }

        pre = matrix_dcsr(pre_rows_pointers, a.rows_gpu(), pre_cols_indices_gpu,
                          a.nrows(), b_cols, pre_nnz, a.nzr());
    }


    uint32_t get_group(uint32_t size) {
        if (size < 33) return size;
        if (size < 65) return 33;
        if (size < 129) return 34;
        if (size < 257) return 35;
        if (size < 513) return 36;
        return 37;
    }


    void count_workload(Controls &controls,
                        cl::Buffer &nnz_estimation_out,
                        const matrix_dcsr &a,
                        const matrix_dcsr &b) {

        auto count_workload = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                uint32_t, uint32_t>("count_workload", "count_workload");
        count_workload.set_work_size(a.nzr());

        cl::Buffer nnz_estimation;
        CHECK_CL(nnz_estimation = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (a.nzr() + 1)),
                 CLBOOL_CREATE_BUFFER_ERROR, 79569681);

        CHECK_RUN(count_workload.run(controls, nnz_estimation, a.rpt_gpu(), a.cols_gpu(),
                                     b.rows_gpu(), b.rpt_gpu(), a.nzr(), b.nzr()), 2129994)

        nnz_estimation_out = std::move(nnz_estimation);
    }


    void prepare_positions(Controls &controls,
                           cl::Buffer &positions,
                           const cl::Buffer &array,
                           uint32_t size,
                           const std::string &program_name
    ) {
        auto prepare_positions = kernel<cl::Buffer, cl::Buffer, uint32_t>
                ("prepare_positions", program_name)
                .set_work_size(size);

        CHECK_RUN(prepare_positions.run(controls, positions, array, size), 465348324);
    }


    void set_positions(Controls &controls,
                       cl::Buffer &c_rpt,
                       cl::Buffer &c_rows,
                       const cl::Buffer &nnz_estimation,
                       const cl::Buffer &a_rows,
                       const cl::Buffer &positions,
                       uint32_t a_nzr
    ) {
        auto set_positions = kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
                ("set_positions", "set_positions_pointers_and_rows");
        set_positions.set_work_size(a_nzr);

        CHECK_RUN(set_positions.run(controls, c_rpt, c_rows,
                                    nnz_estimation, a_rows, positions,
                                    a_nzr).wait(), 325441120);
    }
}