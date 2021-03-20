//#ifndef RUN
//
//#include "../clion_defines.cl"
//#define GROUP_SIZE 256
//#define restrict
//#define local
//#endif

#define __local local

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void update_pref_sum(__global uint* restrict pref_sum,
                              __global const uint* restrict vertices,
                              uint n,
                              uint leaf_size) {

    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    uint block_size = get_local_size(0);
    uint leaves_in_crown = block_size;

    uint global_leaf_id = global_id / leaf_size;
    uint local_leaf_id = global_leaf_id & (leaves_in_crown - 1);


    if (local_leaf_id == 0 || global_id >= n) return;
//    if (leaf_size == 256 && global_leaf_id == 1 && get_local_id(0) == 0) {
//        printf("vertices[global_leaf_id - 1]: %d\n", vertices[global_leaf_id - 1]);
//    }
    pref_sum[global_id] += vertices[global_leaf_id];
}