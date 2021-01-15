//#include "clion_defines.cl"
//#define GROUP_SIZE 256


/*
 * what we have:                    1 3 5 5 7 8 10 11 11 1 314 14 14 17
 * what we want:                    1 1 1 0 1 1 1  1  0  1  1  0  0  1
 * so later the positions will be:  0 1 2 2 3 4 5  6  6  7  8  8  8  9
 * apply exclusive prefix sum on it, total sum == size of new matrix in another variable
 *
 *
 */


__kernel void prepare_array_for_positions(__global unsigned int* result,
                                          __global const unsigned int* rows,
                                          __global const unsigned int* cols,
                                          unsigned int size
                                          ) {

    unsigned int global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    // if on global_id - 1 we have the same value, we write 0 in result, otherwise 1
    result[global_id] = global_id == 0 ? 1 :
                        (cols[global_id] == cols[global_id - 1]) && (rows[global_id] == rows[global_id - 1]) ?
                        0 : 1;
}


__kernel void prepare_array_for_rows_positions(__global unsigned int* result,
                                          __global const unsigned int* rows,
                                          unsigned int size
) {

    unsigned int global_id = get_global_id(0);

    if (global_id >= size) {
        return;
    }

    // if on global_id - 1 we have the same value, we write 0 in result, otherwise 1
    result[global_id] = global_id == 0 ? global_id : (rows[global_id] == rows[global_id - 1]) ?
                        0 : 1;

}



__kernel void prepare_array_for_shift(__global unsigned int* result,
                                      __global const unsigned int* rows,
                                      __global const unsigned int* cols,
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
