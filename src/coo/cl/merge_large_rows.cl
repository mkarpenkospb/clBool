#include "clion_defines.cl"

#define SWAP(a,b) {__local uint * tmp=a; a=b; b=tmp;}

#define GROUP_SIZE 256
#define BUFFER_SIZE 256
// we want to generate code for 31 different heap sizes, and we'll send this
// constant as a compilation parameter

inline uint
search_global(__global const unsigned int *array, uint value, uint size) {
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


inline uint
search_local(__local const unsigned int *array, uint value, uint size) {
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

// а не сделать ли scan маленьким
// TODO: подогнать под ближайшую степень двойки
inline void
scan(__local uint *positions) {
    uint group_size = get_local_size(0);
    uint local_id = get_local_id(0);
    uint dp = 1;

    for(uint s = group_size>>1; s > 0; s >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(local_id < s)
        {
            uint i = dp*(2 * local_id + 1) - 1;
            uint j = dp*(2 * local_id + 2) - 1;
            positions[j] += positions[i];
        }

        dp <<= 1;
    }

    if(local_id == BUFFER_SIZE - 1) {
        positions[BUFFER_SIZE] = positions[local_id];
        positions[local_id] = 0;
    }

    for(uint s = 1; s < group_size; s <<= 1)
    {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id < s)
        {
            uint i = dp*(2 * local_id + 1) - 1;
            uint j = dp*(2 * local_id + 2) - 1;

            unsigned int t = positions[j];
            positions[j] += positions[i];
            positions[i] = t;
        }
    }
}


inline void
set_positions(__local const uint *positions, __local const uint *vals, uint size, __local uint *result) {
    uint local_id = get_local_id(0);

    if (local_id >= size) return;
    if (local_id == size - 1 && positions[local_id] != positions[BUFFER_SIZE]) {
        result[positions[local_id]] = vals[local_id];
    }
    if (positions[local_id] != positions[local_id + 1]) {
        result[positions[local_id]] = vals[local_id];
    }
}

inline uint ceil_to_power2(uint v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}


__kernel void local_merge(__global const unsigned int *indices,
                          unsigned int group_start, // indices_pointers[workload_group_id]
                          unsigned int group_length,

                         __global const unsigned int *pre_matrix_rows_pointers,
                         __global unsigned int *pre_matrix_cols_indices,

                         __global unsigned int *nnz_estimation,

                         __global const unsigned int *a_rows_pointers,
                         __global const unsigned int *a_cols,

                         __global const unsigned int *b_rows_pointers,
                         __global const unsigned int *b_rows_compressed,
                         __global const unsigned int *b_cols,
                         const unsigned int b_nzr
) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);

    // не получится сливать массивы inplace
    __local uint merge_buffer1[BUFFER_SIZE];
    __local uint merge_buffer2[BUFFER_SIZE];

    // буферы придется свапать, для этого указатели.
    __local uint* buff_to_copy = merge_buffer1;
    __local uint* buff_to_merge = merge_buffer2;
    __local uint* tmp;

    // скан для определения места элемента. +1 так как это exclusive scan, и в самый последний элемент мы запищем сумму
    __local uint positions[BUFFER_SIZE + 1];

    uint fill_pointer = 0;

    uint row_index = indices[group_start + group_id];
    uint a_start = a_rows_pointers[row_index];
    uint a_end = a_rows_pointers[row_index + 1];

    for (uint a_row_pointer = a_start; a_row_pointer < a_end; ++a_row_pointer) {

        uint col_index = a_cols[a_row_pointer];
        uint b_row_pointer = search_global(b_rows_compressed, col_index, b_nzr);

        if (b_row_pointer == b_nzr) continue;

        uint b_start = b_rows_pointers[b_row_pointer];
        uint b_end = b_rows_pointers[b_row_pointer + 1];
        uint b_row_length = b_end - b_start;
        // сколько понадобится шагов со страйдом в group_size, их будет не более двух! Так как выделим половину

        uint steps = (b_row_length + GROUP_SIZE - 1) / GROUP_SIZE;

        for (uint group_step = 0; group_step < steps; ++group_step) {

            uint elem_id_local = group_step * GROUP_SIZE + local_id;
            uint elem_id = b_start + elem_id_local;

            if (elem_id < b_end) {
                uint fill_position = elem_id_local + fill_pointer;
                if (fill_position < BUFFER_SIZE)  {
                    // самый первый ряд просто копируем в начальный буфер
                    if (a_row_pointer == a_start) {
                        buff_to_merge[fill_position] = b_cols[elem_id];
                        continue;
                    }

                    buff_to_copy[fill_position] = b_cols[elem_id];
                    // теперь надо бинпоиском проверить, нет ли этого элемента среди уже имеющихся
                    positions[elem_id_local] = search_local(buff_to_copy, buff_to_copy[fill_position], fill_pointer) == fill_pointer ? 1 : 0;
                } else {
                    // не помещаемся в буфер, копируем что есть и выходим
                }
            }
        }

        // посчитать преф суммы на positions, сначала обнулим все что выходим за пределы
        if (local_id >= b_row_length) positions[local_id] = 0;
        scan(positions);
        set_positions(positions, buff_to_copy + fill_pointer, b_row_length, buff_to_merge + fill_pointer);
        // в том же буфере, куда копировали, избавимся от существующих до этого элементов


        fill_pointer += b_row_length;



    }
}