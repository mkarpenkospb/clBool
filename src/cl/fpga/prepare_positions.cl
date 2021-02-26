//#ifndef RUN
//
//#include "../clion_defines.cl"
//#define GROUP_SIZE 256
//#define restrict
//#define local
//#endif

#define __local local

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void prepare_array_for_positions(__global uint* restrict result,
                                          __global const uint* restrict rows,
                                          __global const uint* restrict cols,
                                          uint size
                                          ) {

    uint global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    // if on global_id - 1 we have the same value, we write 0 in result, otherwise 1
    result[global_id] = global_id == 0 ? 1 :
                        (cols[global_id] == cols[global_id - 1]) && (rows[global_id] == rows[global_id - 1]) ?
                        0 : 1;
}

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void prepare_array_for_rows_positions(__global uint* restrict result,
                                               __global const uint* restrict rows,
                                               uint size
) {

    uint global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    // if on global_id - 1 we have the same value, we write 0 in result, otherwise 1
    result[global_id] = global_id == 0 ? 1 : (rows[global_id] == rows[global_id - 1]) ?
                        0 : 1;

}


__attribute__((reqd_work_group_size(256,1,1)))
__kernel void prepare_array_for_shift(__global uint* restrict result,
                                      __global const uint* restrict rows,
                                      __global const uint* restrict cols,
                                      uint size
                                      ) {

    uint global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    // if on global_id - 1 we have the same value, we write 1 in result,
    // otherwise 0
    result[global_id] = global_id == 0 ? global_id :
                        (cols[global_id] == cols[global_id - 1]) && (rows[global_id] == rows[global_id - 1]) ?
                        1 : 0;
}

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