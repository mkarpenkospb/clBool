#define __local local

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
