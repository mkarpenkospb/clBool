#ifndef RUN

#include "../clion_defines.cl"

#define GROUP_SIZE 32
#define NNZ_ESTIMATION 32

#endif
#define TABLE_SIZE 32
// 4 threads for 4 roes
#define PWARP 4
// how many rows (tables) can wo process by one threadblock
#define ROWS_PER_TB (GROUP_SIZE / PWARP)
#define HASH_SCAL 107

uint search_global(__global const unsigned int *array, uint value, uint size) {
    uint l = 0;
    uint r = size;
    uint m = l + ((r - l) / 2);
    while (l < r) {
        if (array[m] == value) {
            return m;
        }

        if (array[m] < value) {
            l = m + 1;
        } else {
            r = m;
        }

        m = l + ((r - l) / 2);
    }

    return size;
}

__kernel void hash_symbolic_pwarp(__global const unsigned int *indices,
                                  unsigned int group_start,
                                  unsigned int group_length,

                                  __global
                                  const unsigned int *pre_matrix_rows_pointers, // указатели, куда записывать, или преф сумма по nnz_estimation
                                  __global unsigned int *pre_matrix_cols_indices, // указатели сюда, записываем сюда

                                  __global unsigned int *nnz_estimation, // это нужно обновлять

                                  __global const unsigned int *a_rows_pointers,
                                  __global const unsigned int *a_cols,

                                  __global const unsigned int *b_rows_pointers,
                                  __global const unsigned int *b_rows_compressed,
                                  __global const unsigned int *b_cols,
                                  const unsigned int b_nzr
) {

    uint hash, old, row_index, a_start, a_end, col_index, b_col;

    uint row_id_bin = get_global_id(0) / PWARP;
    uint local_row_id = row_id_bin & (ROWS_PER_TB - 1); // row_id_bin & (ROWS_PER_TB - 1) == row_id_bin % ROWS_PER_TB
    uint thread_id = get_global_id(0) & (PWARP - 1); // get_global_id(0) & (PWARP - 1) == get_global_id(0) % PWARP
    uint row_pos = group_start + row_id_bin; // row for pwarp

    __local uint hash_table[ROWS_PER_TB * TABLE_SIZE];
    __local uint nz_count[ROWS_PER_TB * PWARP];
    __local uint *thread_nz = nz_count + (PWARP * local_row_id + thread_id);
    thread_nz[0] = 0;
    __local uint *local_table = hash_table + (TABLE_SIZE * local_row_id);
    // init hash_table

    for (uint i = thread_id; i < TABLE_SIZE; i += PWARP) {
        local_table[i] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // if (pwarp_id >= group_start + group_length) return; -- cannot return because of barrier later
    if (row_pos < group_start + group_length) {
        row_index = indices[row_pos];
        a_start = a_rows_pointers[row_index];
        a_end = a_rows_pointers[row_index + 1];

        for (uint a_prt = a_start + thread_id; a_prt < a_end; a_prt += PWARP) {
            col_index = a_cols[a_prt]; // позицию этого будем искать в матрице B
            b_col = search_global(b_rows_compressed, col_index, b_nzr);

            // Now go to hashtable and search for b_col
            hash = (b_col * HASH_SCAL) & (TABLE_SIZE - 1);
            while (true) {
                if (local_table[hash] == b_col) {
                    break;
                }
                else if (local_table[hash] == -1) {
                    old = atom_cmpxchg(local_table + hash, -1, b_col);
                    if (old == -1) {
                        thread_nz[0]++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (TABLE_SIZE - 1);
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (row_pos >= group_start + group_length) return;


    if (thread_id == 0) {
        nnz_estimation[0] = thread_nz[0] + thread_nz[1] + thread_nz[2] + thread_nz[3];
    }
}