//#include "clion_defines.cl"
//#define GROUP_SIZE 256


/*
 *
 *
 * we have positions:               0 1 2 2 3 4 5  6  6  7  8  8  0  9
 * each position "If my closer index that less than me is the same, I'm not moving"
 *
 */


__kernel void set_positions(__global unsigned int* newRows,
                            __global unsigned int* newCols,
                            __global const unsigned int* rows,
                            __global const unsigned int* cols,
                            __global const unsigned int* positions,
                            unsigned int n
                            ) {

    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int global_id = get_global_id(0);

    if (global_id == 0) {
        newRows[global_id] = rows[global_id];
        newCols[global_id] = cols[global_id];
        return;
    }

    if (global_id >= n) return;

    if (positions[global_id] != positions[global_id - 1]) {
        newRows[positions[global_id]] = rows[global_id];
        newCols[positions[global_id]] = cols[global_id];
    }
}


__kernel void set_positions_rows(__global unsigned int* newRows,
                            __global const unsigned int* rows,
                            __global const unsigned int* positions,
                            unsigned int n
) {

    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int global_id = get_global_id(0);

    if (global_id == 0) {
        newRows[global_id] = rows[global_id];
        return;
    }

    if (global_id >= n) return;

    if (positions[global_id] != positions[global_id - 1]) {
        newRows[positions[global_id]] = rows[global_id];
    }
}