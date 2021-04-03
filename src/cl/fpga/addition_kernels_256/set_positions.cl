#define GROUP_SIZE 256

__attribute__((reqd_work_group_size(GROUP_SIZE,1,1)))
__kernel void set_positions(__global uint* restrict newRows,
                            __global uint* restrict newCols,
                            __global const uint* restrict rows,
                            __global const uint* restrict cols,
                            __global const uint* restrict positions,
                            uint size
                            ) {
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