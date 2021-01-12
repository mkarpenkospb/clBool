//#include "clion_defines.cl"
//
//#define GROUP_SIZE 256
//#define NNZ_ESTIMATION 32

uint search_global(__global const unsigned int *array, uint value, uint size) {
    uint l = 0;
    uint r = size;
    uint m = l + ((r - l) / 2);
    while (l + 1 < r) {
        if (array[m] == value) {
            return m;
        }

        if (array[m] < value) {
            l = m;
        } else {
            r = m;
        }

        m = l + ((r - l) / 2);
    }

    return size;
}

/*
 * indices -- array of indices to work with
 * group_start, group_length - start and length of the group with nnz-estimation of NNZ_ESTIMATION
 *
 */

__kernel void bitonic_esc(__global const unsigned int *indices,
                         unsigned int group_start,
                         unsigned int group_length,

                         __global const unsigned int *a_rows_pointers,
                         __global const unsigned int *a_cols,

                         __global const unsigned int *b_rows_pointers,
                         __global const unsigned int *b_rows_compressed,
                         __global const unsigned int *b_cols,
                         const unsigned int b_nzr
) {

    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint row_pos = group_start + group_id;
    uint group_size = get_local_size(0);


    if (row_pos >= group_start + group_length) return;

    __local uint cols[NNZ_ESTIMATION];
    // ------------------ fill cols  -------------------

    /*
     * fill each row in parallel
     */

    uint row_index = indices[row_pos];

    uint a_start = a_rows_pointers[row_index];
    uint a_end = a_rows_pointers[row_index + 1];
    uint heap_fill_pointer = 0;

    for (uint a_pointer = a_start; a_pointer < a_end; ++a_pointer) {
        uint col_index = a_cols[a_pointer];
        uint b_row_pointer = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_row_pointer == b_nzr) continue;

        uint b_start = b_rows_pointers[b_row_pointer];
        uint b_end = b_rows_pointers[b_row_pointer + 1];

        uint b_row_length = b_end - b_start;
        uint steps = (b_row_length + group_size - 1) / group_size;

        for (uint group_step = 0; group_step < steps; ++group_step) {
            uint elem_id = group_step * group_size + local_id;
            if (elem_id < b_row_length) {
                cols[heap_fill_pointer + elem_id] = b_cols[elem_id];
            }
        }
        heap_fill_pointer += b_row_length;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // ---------------------- bitonic sort -------------------

    uint segment_length = 2;

    /*
     * nearest power of 2, not less than array size
     */
    uint outer = pow(2, ceil(log2((float) NNZ_ESTIMATION)));

    while (outer != 1) {
        uint half_segment_length = segment_length / 2;
        uint local_line_id = local_id % half_segment_length;
        uint local_twin_id = segment_length - local_line_id - 1;
        uint group_line_id = local_id / half_segment_length;
        uint line_id = segment_length * group_line_id + local_line_id;
        uint twin_id = segment_length * group_line_id + local_twin_id;

        if (line_id < NNZ_ESTIMATION && twin_id < NNZ_ESTIMATION && cols[line_id] > cols[twin_id]) {
            uint tmp = cols[line_id];
            cols[line_id] = cols[twin_id];
            cols[twin_id] = tmp;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint j = half_segment_length; j > 1; j >>= 1) {
            uint half_j = j / 2;
            local_line_id = local_id % half_j;
            local_twin_id = local_line_id + half_j;
            group_line_id = local_id / half_j;
            line_id = j * group_line_id + local_line_id;
            twin_id = j * group_line_id + local_twin_id;

            if (line_id < NNZ_ESTIMATION && twin_id < NNZ_ESTIMATION && cols[line_id] > cols[twin_id]) {
                uint tmp = cols[line_id];
                cols[line_id] = cols[twin_id];
                cols[twin_id] = tmp;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        outer >>= 1;
        segment_length <<= 1;
    }



    // -------------------------------------- scan -----------------------------------------------------

    __local uint positions[NNZ_ESTIMATION];

    if (local_id < NNZ_ESTIMATION) {
        positions[local_id] = local_id == 0 ? 1 :
                              (cols[local_id] == cols[local_id - 1])  ? 0 : 1;
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    uint dp = 1;
    /*
     * exclusive / inclusive doesn't matter -- first element is 0
     */
    for(uint s = group_size>>1; s > 0; s >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_id < s)
        {
            uint i = dp*(2 * local_id + 1) - 1;
            uint j = dp*(2 * local_id + 2) - 1;
            positions[j] += positions[i];
        }

        dp <<= 1;
    }


    if(local_id == NNZ_ESTIMATION - 1) {
        /* !!!!!!!!!!!! В ЭТОМ МЕСТЕ РАЗМЕР МАССИВА */
        positions[local_id] = 0;
    }

    for(uint s = 1; s < group_size; s <<= 1)
    {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id < s)
        {
            uint i = dp*(2 * local_id + 1) - 1;
            uint j = dp*(2 * local_id + 2) - 1;

            unsigned int t = positions[j];
            positions[j] += positions[i];
            positions[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    /*
     * вывести в глобальную память
     */

}
