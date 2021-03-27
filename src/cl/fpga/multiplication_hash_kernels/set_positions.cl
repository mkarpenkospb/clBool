#define __local local

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void set_positions(__global uint* restrict newRows,
                            __global uint* restrict newCols,
                            __global const uint* restrict rows,
                            __global const uint* restrict cols,
                            __global const uint* restrict positions,
                            uint size
                            ) {

    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint global_id = get_global_id(0);

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

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void set_positions_pointers_and_rows(__global uint* restrict newRowsPosition,
                                              __global uint* restrict newRows,
                                              __global const uint* restrict rowsPositions,
                                              __global const uint* restrict rows,
                                              __global const uint* restrict positions,
                                              uint nnz, // old nzr
                                              uint old_nzr,
                                              uint new_nzr
) {
    uint global_id = get_global_id(0);

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

__attribute__((reqd_work_group_size(256,1,1)))
__kernel void set_positions_rows(__global uint* restrict rows_pointers,
                                 __global uint* restrict rows_compressed,
                                 __global const uint* restrict rows,
                                 __global const uint* restrict positions,
                                 uint size,
                                 uint nzr
) {
    uint global_id = get_global_id(0);

    if (global_id == size - 1) {
        if (positions[global_id] != size) {
            rows_pointers[positions[global_id]] = global_id;
            rows_compressed[positions[global_id]] = rows[global_id];
        }
        rows_pointers[nzr] = size;
        return;
    }

    if (global_id >= size) return;

    if (positions[global_id] != positions[global_id + 1]) {
        rows_pointers[positions[global_id]] = global_id;
        rows_compressed[positions[global_id]] = rows[global_id];
    }
}