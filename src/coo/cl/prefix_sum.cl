#include "clion_defines.cl"
//
#define SWAP(a,b) {__local unsigned int * tmp=a; a=b; b=tmp;}
//#define GROUP_SIZE 256
// count prefixes itself on input array
// and vertices as thr output

// TODO: optimise bank conflicts and num of threads, we need two times less
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
__kernel void scan_blelloch(
        __global unsigned int * vertices,
        __global unsigned int * pref_sum,
        __local unsigned int * tmp,
        unsigned int n)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint block_size = get_local_size(0);
    uint dp = 1;

    tmp[local_id] = global_id < n ? pref_sum[global_id] : 0;

    for(uint s = block_size>>1; s > 0; s >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_id < s)
        {
            uint i = dp*(2 * local_id + 1) - 1;
            uint j = dp*(2 * local_id + 2) - 1;
            tmp[j] += tmp[i];
        }

        dp <<= 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id == block_size - 1) {
        if (global_id < n) {
            pref_sum[global_id] = tmp[local_id];
        }
        vertices[group_id] = tmp[local_id];
        tmp[local_id] = 0;
    }

    for(uint s = 1; s < block_size; s <<= 1)
    {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id < s)
        {
            uint i = dp*(2 * local_id + 1) - 1;
            uint j = dp*(2 * local_id + 2) - 1;

            unsigned int t = tmp[j];
            tmp[j] += tmp[i];
            tmp[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id != 0 && (global_id - 1) <= n) {
        pref_sum[global_id - 1] = tmp[local_id];
    }
}

__kernel void update_pref_sum(__global unsigned int * pref_sum,
                              __global const unsigned int * vertices,
                              unsigned int n,
                              unsigned int leaf_size) {

    uint global_id = get_global_id(0);
    if (global_id >= n) return;

    uint block_size = get_local_size(0);
    uint leaves_in_crown = block_size;

    uint global_leaf_id = global_id / leaf_size;
    uint local_leaf_id = global_leaf_id % leaves_in_crown;


    if (local_leaf_id == 0 || global_id >= n) return;

    pref_sum[global_id] += vertices[global_leaf_id - 1];

}