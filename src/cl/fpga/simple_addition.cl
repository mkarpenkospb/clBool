#ifndef RUN

#include "../clion_defines.cl"
#define GROUP_SIZE 256

#endif

#define GROUP_SIZE 256

__kernel void add(__global uint *c, __global uint *a, __global uint *b, uint size_a, uint size_b, uint size_c) {
    uint global_id = get_global_id(0);
    if (global_id >= size_c) return;
    if (global_id < size_a && global_id < size_b) {
        c[global_id] = a[global_id] + b[global_id];
        return;
    }
    if (global_id < size_a) {
        c[global_id] = a[global_id];
        return;
    }
    if (global_id < size_b) {
        c[global_id] = b[global_id];
    }
}

