#include <iostream>
#include "controls.hpp"
#include "program.hpp"
#include "utils.hpp"
using namespace utils;

int main() {
    std::cout <<"Hello world\n";
    Controls controls = Controls(); //create_controls();
    std::cout <<"success\n";
//    uint32_t n = controls.block_size * 5000;
//    cpu_buffer_f a(n);
//    cpu_buffer_f b(n);
//    cpu_buffer_f c(n);
//    fill_random_buffer(a, 165131233);
//    fill_random_buffer(b, 176713);
//    print_cpu_buffer(a, 10);
//    print_cpu_buffer(b, 10);
//
//    cl::Buffer a_gpu(controls.queue, a.begin(), a.end(), false);
//    cl::Buffer b_gpu(controls.queue, b.begin(), b.end(), false);
//    cl::Buffer c_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(cpu_buffer_f::value_type) * c.size());
//
//    auto p = program<cl::Buffer, cl::Buffer, cl::Buffer, cl_uint>("simple_addition")
//            .set_kernel_name("aplusb")
//            .set_needed_work_size(n);
//
//
//    for (uint32_t i = 0; i < n; ++i) {
//        c[i] = a[i] + b[i];
//    }
//    p.run(controls, a_gpu, b_gpu, c_gpu, n);
//
//    compare_buffers(controls, c_gpu, c, n);

//    std::cout  << clP.getInfo<CL_PROGRAM_NUM_DEVICES>() << std::endl;
}