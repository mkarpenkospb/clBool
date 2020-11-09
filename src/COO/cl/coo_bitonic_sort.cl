#include "clion_defines.cl"

#define GROUP_SIZE 256

//void swap(unsigned int* a, unsigned int* b) {
//    unsigned int
//}

bool is_greater_local(__local const unsigned int* rows,
                      __local const unsigned int* cols,
                      unsigned int line_id,
                      unsigned int twin_id) {
    if (rows[line_id] > rows[twin_id]) {
        return true;
    }
    return cols[line_id] > cols[twin_id];
}


bool is_greater_global(__global const unsigned int* rows,
                       __global const unsigned int* cols,
                        unsigned int line_id,
                        unsigned int twin_id) {
    if (rows[line_id] > rows[twin_id]) {
        return true;
    }
    return cols[line_id] > cols[twin_id];
}

__kernel void local_bitonic_begin(__global unsigned int* rows, __global unsigned int* cols) {

    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int work_size = GROUP_SIZE * 2;

    __local unsigned int local_rows[GROUP_SIZE * 2];
    __local unsigned int local_cols[GROUP_SIZE * 2];

    unsigned int tmp_row = 0;
    unsigned int tmp_col = 0;

    local_cols[local_id] = cols[work_size * group_id + local_id];
    local_cols[local_id + GROUP_SIZE] = rows[work_size * group_id + local_id + GROUP_SIZE];

    local_rows[local_id] = rows[work_size * group_id + local_id];
    local_rows[local_id + GROUP_SIZE] = rows[work_size * group_id + local_id + GROUP_SIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int outer = work_size;
    unsigned int segment_length = 2;
    while (outer != 1) {
        unsigned int local_line_id = local_id % (segment_length / 2);
        unsigned int local_twin_id = segment_length - local_line_id - 1;
        unsigned int group_line_id = local_id / (segment_length / 2);
        unsigned int line_id = segment_length * group_line_id + local_line_id;
        unsigned int twin_id = segment_length * group_line_id + local_twin_id;

        if (is_greater_local(local_rows, local_cols, line_id, twin_id)) {
            tmp_row = local_rows[line_id];
            tmp_col = local_cols[line_id];


            local_rows[line_id] = local_rows[twin_id];
            local_cols[line_id] = local_cols[twin_id];

            local_rows[twin_id] = tmp_row;
            local_cols[twin_id] = tmp_col;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int j = segment_length / 2; j > 1; j >>= 1) {
            local_line_id = local_id % (j / 2);
            local_twin_id = local_line_id + (j / 2);
            group_line_id = local_id / (j / 2);
            line_id = j * group_line_id + local_line_id;
            twin_id = j * group_line_id + local_twin_id;
            if (is_greater_local(local_rows, local_cols, line_id, twin_id)) {
                tmp_row = local_rows[line_id];
                tmp_col = local_cols[line_id];

                local_rows[line_id] = local_rows[twin_id];
                local_cols[line_id] = local_cols[twin_id];

                local_rows[twin_id] = tmp_row;
                local_cols[twin_id] = tmp_col;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        outer >>= 1;
        segment_length <<= 1;
    }

    cols[work_size * group_id + local_id] =  local_cols[local_id];
    cols[work_size * group_id + local_id + GROUP_SIZE] = local_cols[local_id + GROUP_SIZE];

    rows[work_size * group_id + local_id] =  local_rows[local_id];
    rows[work_size * group_id + local_id + GROUP_SIZE] = local_rows[local_id + GROUP_SIZE];
}


__kernel void bitonic_global_step(__global unsigned int* rows,
                                  __global unsigned int* cols,
                                  unsigned int segment_length,
                                  unsigned int mirror)
{
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_line_id = global_id % (segment_length / 2);
    unsigned int local_twin_id = mirror ? segment_length - local_line_id - 1 : local_line_id + (segment_length / 2);
    unsigned int group_line_id = global_id / (segment_length / 2);
    unsigned int line_id = segment_length * group_line_id + local_line_id;
    unsigned int twin_id = segment_length * group_line_id + local_twin_id;

    unsigned int tmp_row = 0;
    unsigned int tmp_col = 0;

    if (is_greater_global(rows, cols, line_id, twin_id)) {
        tmp_row = rows[line_id];
        tmp_col = cols[line_id];

        rows[line_id] = rows[twin_id];
        cols[line_id] = cols[twin_id];

        rows[twin_id] = tmp_row;
        rows[twin_id] = tmp_col;
    }
}

__kernel void bitonic_local_endings(__global unsigned int* rows,
                                    __global unsigned int* cols)
{
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int work_size = GROUP_SIZE * 2;

    __local unsigned int local_rows[GROUP_SIZE * 2];
    __local unsigned int local_cols[GROUP_SIZE * 2];

    unsigned int tmp_row = 0;
    unsigned int tmp_col = 0;

    local_cols[local_id] = cols[work_size * group_id + local_id];
    local_cols[local_id + GROUP_SIZE] = rows[work_size * group_id + local_id + GROUP_SIZE];

    local_rows[local_id] = rows[work_size * group_id + local_id];
    local_rows[local_id + GROUP_SIZE] = rows[work_size * group_id + local_id + GROUP_SIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int segment_length = work_size;

    for (unsigned int j = segment_length; j > 1; j >>= 1) {
        unsigned int local_line_id = local_id % (j / 2);
        unsigned int local_twin_id = local_line_id + (j / 2);
        unsigned int group_line_id = local_id / (j / 2);
        unsigned int line_id = j * group_line_id + local_line_id;
        unsigned int twin_id = j * group_line_id + local_twin_id;

        if (is_greater_local(local_rows, local_cols, line_id, twin_id)) {
            tmp_row = local_rows[line_id];
            tmp_col = local_cols[line_id];


            local_rows[line_id] = local_rows[twin_id];
            local_cols[line_id] = local_cols[twin_id];

            local_rows[twin_id] = tmp_row;
            local_cols[twin_id] = tmp_col;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    rows[work_size * group_id + local_id] =  local_rows[local_id];
    rows[work_size * group_id + local_id + GROUP_SIZE] = local_rows[local_id + GROUP_SIZE];
    cols[work_size * group_id + local_id] =  local_cols[local_id];
    cols[work_size * group_id + local_id + GROUP_SIZE] = local_cols[local_id + GROUP_SIZE];
}



