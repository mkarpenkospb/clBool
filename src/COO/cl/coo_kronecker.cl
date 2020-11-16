#include "clion_defines.cl"
//
#define GROUP_SIZE 256

//void swap(unsigned int* a, unsigned int* b) {
//    unsigned int
//}

__kernel void kronecker(__global unsigned int* rowsA,
                        __global unsigned int* colsA,
                        __global unsigned int* rowsB,
                        __global unsigned int* colsB,
                        __global unsigned int* rowsRes,
                        __global unsigned int* colsRes,

                        unsigned int nnzA,
                        unsigned int nnzB,
                        unsigned int size_A,
                        unsigned int size_B
                        ) {
    unsigned int global_id = get_global_id(0);

    unsigned int block_id = global_id / nnzB;
    unsigned int elem_id = global_id % nnzB;

    unsigned int rowA = rowsA[block_id];
    unsigned int colA = colsA[block_id];

    unsigned int rowB = rowsB[elem_id];
    unsigned int colB = colsB[elem_id];

    rowsRes[global_id] = size_B * rowA + rowB;
    colsRes[global_id] = size_B * colA + colB;
}



