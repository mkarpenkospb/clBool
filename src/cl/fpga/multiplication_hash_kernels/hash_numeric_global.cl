#define __local local
#ifndef GPU
#define GROUP_SIZE 8192
#else
#define GROUP_SIZE 256
#endif
#define WARP 32 // TODO add define for amd to 64, for fpga unknown
#define HASH_SCAL 107

uint ceil_to_power2(uint v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void bitonic_sort(__global uint *data,
                           uint size) {

    uint half_segment_length, local_line_id, local_twin_id, group_line_id, line_id, twin_id;
    uint local_id_real = get_local_id(0);

    uint outer = ceil_to_power2(size);
    uint threads_needed = outer / 2;
    // local_id < outer / 2

    uint segment_length = 2;
    while (outer != 1) {
        for (uint local_id = local_id_real; local_id < threads_needed; local_id += GROUP_SIZE) {
            half_segment_length = segment_length / 2;
            // id inside a segment
            local_line_id = local_id & (half_segment_length - 1);
            // index to compare and swap
            local_twin_id = segment_length - local_line_id - 1;
            // segment id
            group_line_id = local_id / half_segment_length;
            // индексы элементов в массиве
            line_id = segment_length * group_line_id + local_line_id;
            twin_id = segment_length * group_line_id + local_twin_id;

            if (line_id < size && twin_id < size && data[line_id] > data[twin_id]) {
                uint tmp = data[line_id];
                data[line_id] = data[twin_id];
                data[twin_id] = tmp;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (uint j = half_segment_length; j > 1; j >>= 1) {
            for (uint local_id = local_id_real; local_id < threads_needed; local_id += GROUP_SIZE) {
                uint half_j = j / 2;
                local_line_id = local_id & (half_j - 1);
                local_twin_id = local_line_id + half_j;
                group_line_id = local_id / half_j;
                line_id = j * group_line_id + local_line_id;
                twin_id = j * group_line_id + local_twin_id;

                if (line_id < size && twin_id < size && data[line_id] > data[twin_id]) {
                    uint tmp = data[line_id];
                    data[line_id] = data[twin_id];
                    data[twin_id] = tmp;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }
        outer >>= 1;
        segment_length <<= 1;
    }
}


uint search_global(__global const uint *array, uint value, uint size) {
    uint l = 0;
    uint r = size;
    uint m = l + ((r - l) / 2);
    while (l < r) {
        if (array[m] == value) {
            return m;
        }

        if (array[m] < value) {
            l = m + 1;
        } else {
            r = m;
        }

        m = l + ((r - l) / 2);
    }

    return size;
}

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