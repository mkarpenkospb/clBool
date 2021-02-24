#include <iostream>
#include "controls.hpp"
#include "program.hpp"
#include "utils.hpp"
using namespace utils;

int main() {
    std::cout <<"Hello world\n";

    Controls controls = create_controls(); //create_controls();

    uint32_t n = 25355321;

    std::vector<char> buffer(4*n);
    ptrdiff_t start = reinterpret_cast<size_t>(&buffer[0]);
    uint32_t byteAlignment = 8;
    start = ((start >> byteAlignment) + 1) << byteAlignment;
    float *ptrStart = reinterpret_cast<float *>(start);

//    cpu_buffer_f a(n);
//    cpu_buffer_f b(n);
//    cpu_buffer_f c(n);
    cpu_buffer_f *pa = new (ptrStart) cpu_buffer_f(n);
    cpu_buffer_f *pb = new (ptrStart + (n + 10)) cpu_buffer_f(n);
    cpu_buffer_f *pc = new (ptrStart + 2*(n + 10)) cpu_buffer_f(n);

    cpu_buffer_f a = *pa;
    cpu_buffer_f b = *pb;
    cpu_buffer_f c = *pc;
    fill_random_buffer(a, 165131233);
    fill_random_buffer(b, 176713);
    print_cpu_buffer(a, 10);
    print_cpu_buffer(b, 10);

    cl::Buffer a_gpu(controls.queue, a.begin(), a.end(), false);
    cl::Buffer b_gpu(controls.queue, b.begin(), b.end(), false);
    cl::Buffer c_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(cpu_buffer_f::value_type) * c.size());

    auto p = program<cl::Buffer, cl::Buffer, cl::Buffer, cl_uint>("simple_addition_branch")
            .set_kernel_name("aplusb")
            .set_needed_work_size(n);


    for (uint32_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
    p.run(controls, a_gpu, b_gpu, c_gpu, n);

    compare_buffers(controls, c_gpu, c, n);

}