#include "clBool_tests.hpp"
#include "coo.hpp"

using namespace clbool;
using namespace clbool::coo_utils;
using namespace clbool::utils;

bool test_new_merge(Controls& controls, uint32_t size_a, uint32_t size_b) {

    cpu_buffer a_cpu(size_a);
    cpu_buffer b_cpu(size_b);
    cpu_buffer c_cpu;

    fill_random_buffer(a_cpu);
    fill_random_buffer(b_cpu);

    std::sort(a_cpu.begin(), a_cpu.end());
    std::sort(b_cpu.begin(), b_cpu.end());

    std::merge(a_cpu.begin(), a_cpu.end(), b_cpu.begin(), b_cpu.end(),
               std::back_inserter(c_cpu));

    cl::Buffer a_gpu = cl::Buffer(controls.queue, a_cpu.begin(), a_cpu.end(), false);
    cl::Buffer b_gpu = cl::Buffer(controls.queue, b_cpu.begin(), b_cpu.end(), false);
    cl::Buffer c_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_cpu.size());
//            print_cpu_buffer(c_cpu);

    auto new_merge = kernel<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t, uint32_t>
            ("for_test/new_merge", "new_merge_full");
    new_merge.set_work_size(a_cpu.size() + b_cpu.size());
   new_merge.run(controls,
                a_gpu, b_gpu, c_gpu, a_cpu.size(), b_cpu.size());

//            std::cout << "~~~~~~~~~~~~~~\n~~~~~~~~~~~~~~\n";
//            print_gpu_buffer(controls, c_gpu, c_cpu.size());

//            std::cout << c_cpu[349] << std::endl;
    return compare_buffers(controls, c_gpu, c_cpu, c_cpu.size());
}