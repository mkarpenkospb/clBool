#include "clion_defines.cl"
//#define GROUP_SIZE 256


bool is_greater_global(__global const unsigned int *rowsA,
                       __global const unsigned int *colsA,
                       __global const unsigned int *rowsB,
                       __global const unsigned int *colsB,
                       unsigned int indexA,
                       unsigned int indexB) {

    return (rowsA[indexA] > rowsB[indexB]) ||
           ((rowsA[indexA] == rowsB[indexB]) && (colsA[indexA] > colsB[indexB]));

}


__kernel void merge(__global unsigned int *rowsC,
                    __global unsigned int *colsC,
                    __global const unsigned int *rowsA,
                    __global const unsigned int *colsA,
                    __global const unsigned int *rowsB,
                    __global const unsigned int *colsB,
                    unsigned int sizeA,
                    unsigned int sizeB) {

    unsigned int diag_index = get_global_id(0);
    unsigned int res_size = sizeA + sizeB;
    unsigned int min_side = sizeA < sizeB ? sizeA : sizeB;
    unsigned int max_side = res_size - min_side;



    // we can allow it because there are no barriers in the code below
    if (diag_index >= res_size) {
        return;
    }

    unsigned int diag_length = diag_index < min_side ? diag_index + 2 :
                               diag_index < max_side ? min_side + 2 :
                               res_size - diag_index;
//
//    if (diag_index == 1676) {
//        printf("sizeA: %d\n",  sizeA);
//        printf("sizeB: %d\n",  sizeB);
//        printf("min_side: %d\n",  min_side);
//        printf("diag_length: %d\n",  diag_length);
//        printf("res_size: %d\n",  res_size);
//    }

    unsigned r = diag_length;
    unsigned l = 0;
    unsigned int m = 0;

    unsigned int below_idx_a = 0;
    unsigned int below_idx_b = 0;
    unsigned int above_idx_a = 0;
    unsigned int above_idx_b = 0;

    unsigned int above = 0; // значение сравнения справа сверху
    unsigned int below = 0; // значение сравнения слева снизу


    while (true) {
        m = (l + r) / 2;
        below_idx_a = diag_index < sizeA ? diag_index - m + 1 : sizeA - m;
        below_idx_b = diag_index < sizeA ? m - 1 : (diag_index - sizeA) + m;

        above_idx_a = below_idx_a - 1;
        above_idx_b = below_idx_b + 1;

        below = m == 0 ? 1 : is_greater_global(rowsA, colsA, rowsB, colsB, below_idx_a,
                                               below_idx_b); //a[below_idx_a] > b[below_idx_b];
        above = m == diag_length - 1 ? 0 : is_greater_global(rowsA, colsA, rowsB, colsB, above_idx_a, above_idx_b);

//        if (diag_index == 1676) {
//            printf("l: %d, r: %d, m: %d\n", l, r, m);
//            printf("below_idx_a: %d, below_idx_b: %d, above_idx_a: %d, above_idx_b: %d\n",
//                   below_idx_a, below_idx_b, above_idx_a, above_idx_b);
//            if (m != 0 ) {
//                printf("below:     rowsA: %d, colsA: %d, rowsB: %d, colsB: %d\n",
//                       rowsA[below_idx_a], colsA[below_idx_a],
//                       rowsB[below_idx_b], colsB[below_idx_b]);
//            }
//
//            if (m != diag_length - 1 ) {
//                printf("above:     rowsA: %d, colsA: %d, rowsB: %d, colsB: %d\n",
//                       rowsA[above_idx_a], colsA[above_idx_a],
//                       rowsB[above_idx_b], colsB[above_idx_b]);
//            }
//        }

        // success
        if (below != above) {
            if ((diag_index < sizeA) && m == 0) {
                rowsC[diag_index] = rowsA[above_idx_a];
                colsC[diag_index] = colsA[above_idx_a];
                return;
            }
            if ((diag_index < sizeB) && m == diag_length - 1) {
                rowsC[diag_index] = rowsB[below_idx_b];
                colsC[diag_index] = colsB[below_idx_b];
                return;
            }
            // в случаях выше эти индексы лучше вообще не трогать, поэтому не объединяю
            bool is_greater = is_greater_global(rowsA, colsA, rowsB, colsB, above_idx_a, below_idx_b);
//            if (!is_greater) {
//                printf("not greater!\n");
//            }
            rowsC[diag_index] = is_greater ? rowsA[above_idx_a] : rowsB[below_idx_b];
            colsC[diag_index] = is_greater ? colsA[above_idx_a] : colsB[below_idx_b];

//            if (diag_index == 1676) {
//                printf("rowsC[diag_index] : %d, colsC[diag_index] : %d\n", rowsC[diag_index], colsC[diag_index]);
//
//            }

            return;
        }

        if (below) {
            l = m;
        } else {
            r = m;
        }
    }
}



