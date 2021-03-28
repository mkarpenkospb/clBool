
#define __local local

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void prepare_for_shift_empty_rows(__global uint* restrict result,
                                           __global const uint* restrict nnz_estimation,
                                           uint size
) {

    uint global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    result[global_id] = nnz_estimation[global_id] == nnz_estimation[global_id + 1]  ? 0 : 1;
}