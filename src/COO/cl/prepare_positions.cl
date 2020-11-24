#include "clion_defines.cl"
#define GROUP_SIZE 256


/*
 * what we have:                    1 3 5 5 7 8 10 11 11 13 14 14 14 17
 * what we want:                    0 1 2 0 3 4 5  6  0  7  8  0  0  9
 * so later the positions will be:  0 1 2 2 3 4 5  6  6  7  8  8  0  9
 *
 * alternative:                     1 3 5 5 7 8 10 11 11 13 14 14 14 17
 * we may want:                     0 0 0 1 0 0 0  0  0  0  0  1  1  0
 * shift for the right positions:   0 0 0 1 1 1 1  1  1  1  1  2  3  3
 *
 */


__kernel void prepare_array_for_positions(__global unsigned int* result,
                                          __global const unsigned int* cols,
                                          __global const unsigned int* rows,
                                          unsigned int size
                                          ) {

    unsigned int global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    // if on global_id - 1 we have the same value, we write 0 in result,
    // otherwise the position itself
    result[global_id] = global_id == 0 ? global_id :
                        (cols[global_id] == cols[global_id - 1]) && (rows[global_id] == rows[global_id - 1]) ?
                        0 : global_id;
}

__kernel void prepare_array_for_shift(__global unsigned int* result,
                                      __global const unsigned int* cols,
                                      __global const unsigned int* rows,
                                      unsigned int size
                                      ) {

    unsigned int global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    // if on global_id - 1 we have the same value, we write 1 in result,
    // otherwise 0
    result[global_id] = global_id == 0 ? global_id :
                        (cols[global_id] == cols[global_id - 1]) && (rows[global_id] == rows[global_id - 1]) ?
                        1 : 0;
}
