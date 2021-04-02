#define __local local

__kernel void update_pref_sum(__global uint* restrict pref_sum,
                              __global const uint* restrict vertices,
                              uint n,
                              uint leaf_size) {

    uint global_id = get_global_id(0) + leaf_size;
    if (global_id >= n) return;
    uint global_leaf_id = global_id / leaf_size;
    pref_sum[global_id] += vertices[global_leaf_id];
}