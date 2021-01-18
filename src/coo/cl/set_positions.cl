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
                            unsigned int size
                            ) {

    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int global_id = get_global_id(0);

    if (global_id == size - 1 && positions[global_id] != size) {
        newRows[positions[global_id]] = rows[global_id];
        newCols[positions[global_id]] = cols[global_id];
        return;
    }

    if (global_id >= size) return;

    if (positions[global_id] != positions[global_id + 1]) {
        newRows[positions[global_id]] = rows[global_id];
        newCols[positions[global_id]] = cols[global_id];
    }
}


__kernel void set_positions_pointers_and_rows(__global unsigned int* newRowsPosition,
                                              __global unsigned int* newRows,
                                              __global const unsigned int* rowsPositions,
                                              __global const unsigned int* rows,
                                              __global const unsigned int* positions,
                                              unsigned int nnz, // old nzr
                                              unsigned int old_nzr,
                                              unsigned int new_nzr
) {

    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int global_id = get_global_id(0);

    if (global_id >= old_nzr) return;

    if (global_id == old_nzr - 1) {
        if (positions[global_id] != old_nzr) {
            newRowsPosition[positions[global_id]] = rowsPositions[global_id];
            newRows[positions[global_id]] = rows[global_id];
        }
        newRowsPosition[new_nzr] = nnz;
        return;
    }

    if (positions[global_id] != positions[global_id + 1]) {
        newRowsPosition[positions[global_id]] = rowsPositions[global_id];
        newRows[positions[global_id]] = rows[global_id];
    }
}


__kernel void set_positions_rows(__global unsigned int* rows_pointers,
                                 __global unsigned int* rows_compressed,
                                 __global const unsigned int* rows,
                                 __global const unsigned int* positions,
                                 unsigned int size,
                                 unsigned int nzr
) {
    unsigned int global_id = get_global_id(0);

    if (global_id == size - 1) {
        if (positions[global_id] != size) {
            rows_pointers[positions[global_id]] = global_id;
            rows_compressed[positions[global_id]] = rows[global_id];
        }
        rows_pointers[nzr] = size;
        return;
    }

    if (global_id >= size) return;
//    if (global_id == 0) {
//        printf("positions[global_id]: %d\n", positions[global_id] );
//        printf("rows[global_id]: %d\n", rows[global_id] );
//    }
//    if (global_id == 1) {
//        printf("positions[global_id]: %d\n", positions[global_id] );
//        printf("rows[global_id]: %d\n", rows[global_id] );
//    }

    if (positions[global_id] != positions[global_id + 1]) {
        rows_pointers[positions[global_id]] = global_id;
        rows_compressed[positions[global_id]] = rows[global_id];
    }
}