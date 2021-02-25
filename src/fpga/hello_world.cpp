#include <libutils/logger.h>
#include <iostream>
#include "controls.hpp"
#include "program.hpp"
#include "utils.hpp"
#include "libutils/timer.h"
using namespace utils;

int main() {
    double time;

    if (DEBUG_ENABLE) *logger << "start" << " \n";
    std::cout <<"Hello world\n";

    Controls controls = create_controls(); //create_controls();

    uint32_t n = 25000000;

//    std::vector<float> buffer(4*n);
//    uint64_t start = reinterpret_cast<uint64_t>(&buffer[0]);
//    printf("Origin address: [%#010x, %#010x]\n",start, &buffer[4*n - 1]);
//    uint64_t byteAlignment = 16;
//    start = alignBy(start, byteAlignment);
//    float *ptrStart = reinterpret_cast<float *>(start);
//    float *ptrStart2 =reinterpret_cast<float *>(alignBy(start + (n + 10) * sizeof(float), byteAlignment));
//    float *ptrStart3 =reinterpret_cast<float *>(alignBy(start + 2 * (n + 10) * sizeof(float), byteAlignment));
//
//    printf("Start addresses: %#010x, %#010x, %#010x\n", ptrStart, ptrStart2, ptrStart3);

    cpu_buffer_f a(n);
    cpu_buffer_f b(n);
    cpu_buffer_f c(n);
//    cpu_buffer_f *pa = new (ptrStart) cpu_buffer_f(n);
//    cpu_buffer_f *pb = new (ptrStart2) cpu_buffer_f(n);
//    cpu_buffer_f *pc = new (ptrStart3) cpu_buffer_f(n);

//    cpu_buffer_f a = *pa;
//    cpu_buffer_f b = *pb;
//    cpu_buffer_f c = *pc;
    fill_random_buffer(a, 165131233);
    fill_random_buffer(b, 176713);
    print_cpu_buffer(a, 10);
    print_cpu_buffer(b, 10);

    t.restart();
    cl::Buffer a_gpu(controls.queue, a.begin(), a.end(), false);
    cl::Buffer b_gpu(controls.queue, b.begin(), b.end(), false);
    cl::Buffer c_gpu(controls.context, CL_MEM_WRITE_ONLY, sizeof(cpu_buffer_f::value_type) * c.size());
    time = t.elapsed();
    if (DEBUG_ENABLE) *logger << "load data to device in " << time << " \n";

    t.restart();
    auto p = program<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>("simple_addition_branch")
            .set_kernel_name("aplusb")
            .set_needed_work_size(n);
    time = t.elapsed();
    if (DEBUG_ENABLE) *logger << "load program in " << time << " \n";

    t.restart();
    for (uint32_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
    time = t.elapsed();
    if (DEBUG_ENABLE) *logger << "run CPU in " << time << " \n";

    t.restart();
    cl::Event ev = p.run(controls, a_gpu, b_gpu, c_gpu, n);
    ev.wait();
    time = t.elapsed();
    if (DEBUG_ENABLE) *logger << "run DEVICE in " << time << " \n";

    compare_buffers(controls, c_gpu, c, n);
}