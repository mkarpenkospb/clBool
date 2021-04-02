__kernel void prepare_array_for_positions(__global uint* restrict result,
                                          __global const uint* restrict rows,
                                          __global const uint* restrict cols,
                                          uint size
                                          ) {

    uint global_id = get_global_id(0);
    if (global_id >= size) return;
    result[global_id] = global_id == 0 ? 1 :
                        (cols[global_id] == cols[global_id - 1]) && (rows[global_id] == rows[global_id - 1]) ?
                        0 : 1;
}