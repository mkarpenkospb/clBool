#define __local local

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void update_pref_sum(__global uint* restrict pref_sum,
                              __global const uint* restrict vertices,
                              uint n,
                              uint leaf_size) {

    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    uint block_size = get_local_size(0);
    uint global_leaf_id = global_id / leaf_size;
    if (global_id >= n) return;
    pref_sum[global_id] += vertices[global_leaf_id + 1];
}