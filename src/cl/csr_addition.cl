#ifndef RUN

#include "clion_defines.cl"

#define GROUP_SIZE 256

#endif

#define MAX_VAL 4294967295;

// see merge_path_count by Artem Khoroshev

__kernel void addition_symbolic(__global const uint *a_rpt,
                                __global const uint *a_cols,
                                __global const uint *b_rpt,
                                __global const uint *b_cols,
                                __global uint *c_rpt,
                                uint nrows
//                                ,thrust::device_ptr<const T> rows_in_bins

) {
    const uint local_id = get_local_id(0);
    const uint global_id = get_global_id(0);
    if (get_global_id(0) == 0) {
        c_rpt[nrows] = 0;
    }
    if (global_id < nrows) {
        c_rpt[global_id] = 0;
    }

    const uint row = get_group_id(0);
    const uint block_size = GROUP_SIZE;

    const uint global_offset_a = a_rpt[row];
    const uint global_offset_b = b_rpt[row];
    const uint a_row_length = a_rpt[row + 1] - global_offset_a;
    const uint b_row_length = b_rpt[row + 1] - global_offset_b;

    const uint block_count = (a_row_length + b_row_length + block_size - 1) / block_size;

    uint begin_a = 0;
    uint begin_b = 0;

    __local uint raw_a[GROUP_SIZE + 2];
    __local uint raw_b[GROUP_SIZE + 2];
    __local uint res[GROUP_SIZE];

    bool dir = true;
    uint item_from_prev_chank = MAX_VAL;

    for (uint i = 0; i < block_count; i++) {
        __local uint max_x_index;
        __local uint max_y_index;

        uint max_x_index_per_thread = 0;
        uint max_y_index_per_thread = 0;

        uint buf_a_size = min(a_row_length - begin_a, block_size);
        uint buf_b_size = min(b_row_length - begin_b, block_size);

        if (local_id == 0) {
            max_x_index = 0;
            max_y_index = 0;
        }

        for (uint j = local_id; j < block_size + 2; j += block_size) {
            if (j > 0 && j - 1 < buf_a_size) {
                raw_a[j] = a_cols[global_offset_a + j - 1 + begin_a];
            } else {
                raw_a[j] = MAX_VAL;
            }
            if (j > 0 && j - 1 < buf_b_size) {
                raw_b[j] = b_cols[global_offset_b + j - 1 + begin_b];
            } else {
                raw_b[j] = MAX_VAL;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        const uint to_process = min(buf_b_size + buf_a_size, block_size);

        for (uint j = local_id; j < to_process; j += block_size) {
            const uint y = j + 2;
            const uint x = 0;

            uint l = 0;
            uint r = j + 2;

            while (r - l > 1) {
                bool ans = raw_b[y - l - (r - l) / 2] > raw_a[x + l + (r - l) / 2];

                l += (r - l) / 2 * ans;
                r -= (r - l) / 2 * !ans;
            }

            uint ans_x = x + l;
            uint ans_y = y - l;

            if (ans_y == 1 || ans_x == 0) {
                if (ans_y == 1) {
                    res[j] = raw_a[ans_x];
                    max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
                } else {
                    res[j] = raw_b[ans_y - 1];
                    max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
                }
            } else {
                if (raw_b[ans_y - 1] > raw_a[ans_x]) {
                    res[j] = raw_b[ans_y - 1];
                    max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
                } else {
                    res[j] = raw_a[ans_x];
                    max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
                }
            }
        }

        atomic_max(&max_x_index, max_x_index_per_thread);
        atomic_max(&max_y_index, max_y_index_per_thread);

        barrier(CLK_LOCAL_MEM_FENCE);

        uint counter = 0;

        if (dir) {
            for (uint m = local_id; m < to_process; m += block_size) {
                if (m > 0)
                    counter += (res[m] - res[m - 1]) != 0;
                else
                    counter += (res[0] - item_from_prev_chank) != 0;
                item_from_prev_chank = res[m];
            }
        } else {
            for (uint m = block_size - 1 - local_id; m < to_process; m += block_size) {
                if (m > 0)
                    counter += (res[m] - res[m - 1]) != 0;
                else
                    counter += (res[0] - item_from_prev_chank) != 0;
                item_from_prev_chank = res[m];
            }
        }

        dir = !dir;
        // TODO: заменить на дерево
        atomic_add(c_rpt + row, counter);

        begin_a += max_x_index;
        begin_b += max_y_index;

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }
}




void scan_blelloch(__local uint *positions) {

    const uint local_id = get_local_id(0);
    const uint block_size = GROUP_SIZE;
    uint dp = 1;

    for (uint s = block_size >> 1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;
            positions[j] += positions[i];
        }

        dp <<= 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        positions[block_size - 1] = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (uint s = 1; s < block_size; s <<= 1) {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;

            uint t = positions[j];
            positions[j] += positions[i];
            positions[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

}

__kernel void addition_numeric(__global const uint *a_rpt,
                               __global const uint *a_cols,
                               __global const uint *b_rpt,
                               __global const uint *b_cols,
                               __global const uint *c_rpt,
                               __global uint *c_cols
//                                 thrust::device_ptr<const T> rows_in_bins
) {

    const uint local_id = get_local_id(0);
    const uint row = get_group_id(0);

    const uint global_offset_a = a_rpt[row];
    const uint global_offset_b = b_rpt[row];
    const uint a_row_length = a_rpt[row + 1] - global_offset_a;
    const uint b_row_length = b_rpt[row + 1] - global_offset_b;

    uint global_offset_c = c_rpt[row];
    const uint block_size = GROUP_SIZE;

    const uint block_count = (a_row_length + b_row_length + block_size - 1) / block_size;

    uint begin_a = 0;
    uint begin_b = 0;

    __local uint raw_a[GROUP_SIZE + 2];
    __local uint raw_b[GROUP_SIZE + 2];
    __local uint res[GROUP_SIZE + 1];
    if (local_id == 0) {
        res[GROUP_SIZE] = 0;
    }

    bool dir = true;
    uint item_from_prev_chank = MAX_VAL;

    for (uint i = 0; i < block_count; i++) {
        __local uint max_x_index;
        __local uint max_y_index;

        uint max_x_index_per_thread = 0;
        uint max_y_index_per_thread = 0;

        uint buf_a_size = min(a_row_length - begin_a, block_size);
        uint buf_b_size = min(b_row_length - begin_b, block_size);

        if (local_id == 0) {
            max_x_index = 0;
            max_y_index = 0;
        }

        for (uint j = local_id; j < block_size + 2; j += block_size) {
            if (j > 0 && j - 1 < buf_a_size) {
                raw_a[j] = a_cols[global_offset_a + j - 1 + begin_a];
            } else {
                raw_a[j] = MAX_VAL;
            }
            if (j > 0 && j - 1 < buf_b_size) {
                raw_b[j] = b_cols[global_offset_b + j - 1 + begin_b];
            } else {
                raw_b[j] = MAX_VAL;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        const uint to_process = min(buf_b_size + buf_a_size, block_size);

        uint answer = MAX_VAL;

        const uint j = dir ? local_id : block_size - 1 - local_id;

        if (j < to_process) {
            const uint y = j + 2;
            const uint x = 0;

            uint l = 0;
            uint r = j + 2;

            while (r - l > 1) {
                bool ans = raw_b[y - l - (r - l) / 2] > raw_a[x + l + (r - l) / 2];

                l += (r - l) / 2 * ans;
                r -= (r - l) / 2 * !ans;
            }

            uint ans_x = x + l;
            uint ans_y = y - l;

            if (ans_y == 1 || ans_x == 0) {
                if (ans_y == 1) {
                    answer = raw_a[ans_x];
                    max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
                } else {
                    answer = raw_b[ans_y - 1];
                    max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
                }
            } else {
                if (raw_b[ans_y - 1] > raw_a[ans_x]) {
                    answer = raw_b[ans_y - 1];
                    max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
                } else {
                    answer = raw_a[ans_x];
                    max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
                }
            }
        }

        atomic_max(&max_x_index, max_x_index_per_thread);
        atomic_max(&max_y_index, max_y_index_per_thread);

        res[j] = answer;

        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        bool take = j < to_process;
        if (j > 0)
            take = take && (answer - res[j - 1]) != 0;
        else
            take = take && (answer - item_from_prev_chank) != 0;

        item_from_prev_chank = answer;

        barrier(CLK_LOCAL_MEM_FENCE);

        res[j] = take;

        barrier(CLK_LOCAL_MEM_FENCE);

        scan_blelloch(res);

        barrier(CLK_LOCAL_MEM_FENCE);
        if (take) {
            c_cols[global_offset_c + res[j]] = answer;
        }

        global_offset_c += res[block_size];

        dir = !dir;

        begin_a += max_x_index;
        begin_b += max_y_index;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


