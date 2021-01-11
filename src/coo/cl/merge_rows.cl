#include "clion_defines.cl"

#define GROUP_SIZE 256
#define NNZ_ESTIMATION 32
#define LOCAL_ARRAY_SIZE 256

uint search_global(__global const unsigned int *array, uint value, uint size) {
    uint l = 0;
    uint r = size;
    uint m = l + ((r - l) / 2);
    while (l + 1 < r) {
        if (array[m] == value) {
            return m;
        }

        if (array[m] < value) {
            l = m;
        } else {
            r = m;
        }

        m = l + ((r - l) / 2);
    }

    return size;
}


uint search_local(__local const unsigned int *array, uint value, uint size) {
    uint l = 0;
    uint r = size;
    uint m = l + ((r - l) / 2);
    while (l + 1 < r) {
        if (array[m] == value) {
            return m;
        }

        if (array[m] < value) {
            l = m;
        } else {
            r = m;
        }

        m = l + ((r - l) / 2);
    }

    return size;
}


void scan(__local uint *positions, uint size) {
    // -------------------------------------- scan -----------------------------------------------------


    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);

    barrier(CLK_LOCAL_MEM_FENCE);

    uint dp = 1;
    /*
     * exclusive / inclusive doesn't matter -- first element is 0
     */
    for (uint s = group_size >> 1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;
            positions[j] += positions[i];
        }

        dp <<= 1;
    }


    if (local_id == size - 1) {
        /* !!!!!!!!!!!! В ЭТОМ МЕСТЕ РАЗМЕР МАССИВА */
        positions[local_id] = 0;
    }

    for (uint s = 1; s < group_size; s <<= 1) {
        dp >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;

            unsigned int t = positions[j];
            positions[j] += positions[i];
            positions[i] = t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

/*
 * indices -- array of indices to work with
 * group_start, group_length - start and length of the group with nnz-estimation of NNZ_ESTIMATION
 *
 */

__kernel void merge_local(__global const unsigned int *indices,
                         unsigned int group_start,
                         unsigned int group_length,

                         __global const unsigned int *a_rows_pointers,
                         __global const unsigned int *a_cols,

                         __global const unsigned int *b_rows_pointers,
                         __global const unsigned int *b_rows_compressed,
                         __global const unsigned int *b_cols,
                         const unsigned int b_nzr
) {
    /*
     * ряд на группу.
     * group_id -  каким рядом в группе занимается текущая группа потоков
     * group_start - начало текущей группы
     */

    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint row_pos = group_start + group_id;
    uint group_size = get_local_size(0);

    /*
     * вряд ли произойдет выход за границы, так как мы можем точно указать число групп при запуске
     * !!!!!!!!!!!!!!!!!!!!! Так как дальше есть барьеры, такого делать нельзя
     */
//    if (row_pos >= group_start + group_length) return;

    /*
     * читаем, какой ряд соответствует нашей группе. row_index --  не номер ряда, а его позиция в
     * списке указателей на ряд.
     */
    uint row_index = indices[row_pos];

    __local uint final_cols[LOCAL_ARRAY_SIZE];
    __local uint current_cols[GROUP_SIZE];
    __local uint positions[GROUP_SIZE];
    bool first_pass = true;

    /*
     * a_start - где в списке a_cols начинается наш ряд. Мы всё ещё не знаем, что это за ряд,
     * но нам не особо и надо
     */
    uint a_start = a_rows_pointers[row_index];
    uint a_end = a_rows_pointers[row_index + 1];

    uint fill_pointer = 0;

    for (uint a_pointer = a_start; a_pointer < a_end; ++a_pointer) {

        /*
         * будем сливать ряды матрицы B, соответствующие col_index.
         * col_index -- это уже точный номер строки матрицы B.
         * Чтобы понять, какая у него позиция, нужно найти её в b_rows_compressed бинпоиском.
         * b_rows_compressed - перечисление ненулевых рядов.
         */

        uint col_index = a_cols[a_pointer];
        uint b_row_pointer = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_row_pointer == b_nzr) continue;

        /*
         * Смотрим, где в списке столбцов начинается наш ряд, с которым работаем.
         * Будем сливать именно индексы столбцов
         */

        uint b_start = b_rows_pointers[b_row_pointer];
        uint b_end = b_rows_pointers[b_row_pointer + 1];

        uint b_row_length = b_end - b_start;

        /*
         * Будем максимально параллельно записывать ряды в имеющийся размер
         */
        uint steps = (b_row_length + group_size - 1) / group_size;

        for (uint group_step = 0; group_step < steps; ++group_step) {
            uint elem_id = group_step * group_size + local_id;
            uint current_length = min(b_row_length & (group_size - 1),  group_size);
            fill_pointer += current_length;
            if (elem_id < b_row_length) {
                /*
                 * В первом проходе не будет дубликатов, поэтому
                 * сразу записываем в финальный массив
                 */
                if (first_pass) {
                    final_cols[local_id] = b_cols[elem_id];
                    first_pass = false;
                    continue;
                }

            }

            current_cols[local_id] = elem_id < b_row_length ? b_cols[elem_id] : 0;

            /*
             * слияние должно происходить тут, так как заполняется группа целиком или ряд
             */

            barrier(CLK_LOCAL_MEM_FENCE);

            /*
             * для каждого элемента в записанном массиве нужно понять, дублируется он в текцщем массиве или нет
             * Это scan просто, но как и в случае с heap, нужно заполнять начиная с 1, так как belloh
             * это exclusive, что нам и надо.
             */

            if (elem_id < b_row_length) {
                positions[local_id] = local_id == 0 ? 1 :
                                      search_local(final_cols, current_cols[local_id], fill_pointer) == fill_pointer ?
                                      1 : 0;
                scan(positions, current_length);
            }
        }

        fill_pointer += b_row_length;
    }


}

__kernel void merge_global() {

}