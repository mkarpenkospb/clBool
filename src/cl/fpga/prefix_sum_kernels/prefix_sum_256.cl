#define __local local
#define GROUP_SIZE 256

// TODO: optimise bank conflicts
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

__attribute__((reqd_work_group_size(GROUP_SIZE,1,1)))
__kernel void scan_blelloch(
        __global uint* restrict vertices,
        __global uint* restrict pref_sum,
        __global uint* restrict total_sum,
        uint n)
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint block_size = get_local_size(0) * 2;
    uint dp = 1;
    __local uint tmp[GROUP_SIZE * 2];
    tmp[2 * local_id] = (global_id * 2) < n ? pref_sum[global_id * 2] : 0;
    tmp[2 * local_id + 1] = (global_id * 2 + 1) < n ? pref_sum[global_id * 2 + 1] : 0;

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

    if(local_id == 0) {
        vertices[group_id] = tmp[block_size - 1];
        *total_sum = tmp[block_size - 1];
        tmp[block_size - 1] = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for(uint s = 1; s < block_size; s <<= 1)
    {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id < s)
        {
            uint i = dp*(2 * local_id + 1) - 1;
            uint j = dp*(2 * local_id + 2) - 1;

            uint t = tmp[j];
            tmp[j] += tmp[i];
            tmp[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (2 * global_id < n) {
        pref_sum[2 * global_id] = tmp[2 * local_id];
    }

    if (2 * global_id + 1 < n) {
        pref_sum[2 * global_id + 1] = tmp[2 * local_id + 1];
    }
}
