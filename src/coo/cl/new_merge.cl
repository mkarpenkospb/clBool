#include "clion_defines.cl"

#define SWAP(a, b) {__local uint * tmp=a; a=b; b=tmp;}

#define GROUP_SIZE 256
#define BUFFER_SIZE 256


// тут нам нужны оба указателия, поэтому вернем примерно sizeB * above_idx_a + above_idx_b
inline uint
merge_pointer_global(__global const uint *a, __local const uint *b, __local uint *c, uint sizeA, uint sizeB, uint diag_index) {
    unsigned int res_size = sizeA + sizeB;
    unsigned int min_side = sizeA < sizeB ? sizeA : sizeB;
    unsigned int max_side = res_size - min_side;

    unsigned int diag_length = diag_index < min_side ? diag_index + 2 :
                               diag_index < max_side ? min_side + 2 :
                               res_size - diag_index;

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

        below = m == 0 ? 1 : a[below_idx_a] > b[below_idx_b];
        above = m == diag_length - 1 ? 0 : a[above_idx_a] > b[above_idx_b];

        // success
        if (below != above) {
            if ((diag_index < sizeA) && m == 0) {
                return sizeB * above_idx_a + above_idx_b;
            }
            if ((diag_index < sizeB) && m == diag_length - 1) {
                return sizeB * below_idx_a + below_idx_b;
            }
            // в случаях выше эти индексы лучше вообще не трогать, поэтому не объединяю
            return a[above_idx_a] > b[below_idx_b] ? above_idx_a * sizeB + above_idx_b : below_idx_a * sizeB + below_idx_b;
        }

        if (below) {
            l = m;
        } else {
            r = m;
        }
    }
}



__kernel void
merge_global(__global const uint *a, __local const uint *b, __local uint *c, uint sizeA, uint sizeB,
             uint step_length
) {
    uint diag_index = get_local_id(0) * step_length;
    if (diag_index >= sizeA + sizeB) return;
    uint mptr = merge_pointer_global(a, b, c, sizeA, sizeB, diag_index);
    uint a_ptr = mptr % sizeA;
    uint b_ptr = mptr / sizeB;

    for (uint i = 0; i < step_length; ++i) {
        if (a_ptr < sizeA && a[a_ptr] < b[b_ptr] || b_ptr >= sizeB) {
            c[diag_index + i] = a[a_ptr++];
            continue;
        }

        if (b_ptr < sizeB) {
            c[diag_index + i] = b[b_ptr++];
            continue;
        }

        return;
    }
}
