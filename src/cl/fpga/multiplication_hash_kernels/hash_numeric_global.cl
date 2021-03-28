#define __local local
#define GROUP_SIZE 256
#define WARP 32 // TODO add define for amd to 64, for fpga unknown
#define HASH_SCAL 107

__attribute__((reqd_work_group_size(GROUP_SIZE,1,1)))
__kernel void hash_numeric_global(__global const uint * restrict indices, // indices -- aka premutation
                                   uint group_start,

                                   __global
                                   const uint * restrict pre_matrix_rows_pointers,
                                  __global uint * restrict c_cols,

                                  __global uint * restrict hash_table_data,
                                  __global const uint * restrict hash_table_offset

) {
    // all data for large rows is already in a global memory,
    // we only need to copy values to the final matrix
    uint row_pos = group_start + get_group_id(0);
    uint row_index = indices[row_pos];
    uint group_id = get_group_id(0);
    uint row_start = pre_matrix_rows_pointers[row_index];
    uint row_end = pre_matrix_rows_pointers[row_index + 1];
    uint row_length = row_end - row_start;
    if (row_length == 0) return;
    uint group_size = get_local_size(0);
    uint local_id = get_local_id(0);
    __global uint* hash_table = hash_table_data + hash_table_offset[group_id];
    __global uint* current_row = c_cols + row_start;
    for (uint i = local_id; i < row_length; i += group_size) {
        current_row[i] = hash_table[i];
    }
}