#include "clion_defines.cl"

#define SWAP(a, b) {__local uint * tmp=a; a=b; b=tmp;}

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


inline void
scan(__local uint *positions, uint scan_size) {
    uint local_id = get_local_id(0);
    uint dp = 1;

    for (uint s = scan_size >> 1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;
            positions[j] += positions[i];
        }

        dp <<= 1;
    }

    if (local_id == BUFFER_SIZE - 1) {
        positions[BUFFER_SIZE] = positions[local_id];
        positions[local_id] = 0;
    }

    for (uint s = 1; s < scan_size; s <<= 1) {
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
}


inline void
scan_global(__global uint *positions, uint scan_size) {
    uint local_id = get_local_id(0);
    uint dp = 1;

    for (uint s = scan_size >> 1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < s) {
            uint i = dp * (2 * local_id + 1) - 1;
            uint j = dp * (2 * local_id + 2) - 1;
            positions[j] += positions[i];
        }

        dp <<= 1;
    }

    if (local_id == BUFFER_SIZE - 1) {
        positions[BUFFER_SIZE] = positions[local_id];
        positions[local_id] = 0;
    }

    for (uint s = 1; s < scan_size; s <<= 1) {
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
}

inline void
set_positions(__local const uint *positions, __local const uint *vals, uint size, __local uint *result,
              uint scan_size) {
    uint local_id = get_local_id(0);

    if (local_id >= size) return;
    if (local_id == size - 1 && positions[local_id] != positions[scan_size]) {
        result[positions[local_id]] = vals[local_id];
    }
    if (positions[local_id] != positions[local_id + 1]) {
        result[positions[local_id]] = vals[local_id];
    }
}

// a or a + b
inline uint
merge_pointer(__local const uint *a, __local const uint *b, __local uint *c, uint sizeA, uint sizeB, uint diag_index) {
    unsigned int res_size = sizeA + sizeB;
    unsigned int min_side = sizeA < sizeB ? sizeA : sizeB;
    unsigned int max_side = res_size - min_side;

    unsigned int diag_length = diag_index < min_side ? diag_index + 2 :
                               diag_index < max_side ? min_side + 2 :
                               res_size - diag_index;

    // Массив A представляем как ряды, B как столбцы


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
                return above_idx_a;
            }
            if ((diag_index < sizeB) && m == diag_length - 1) {
                return sizeA + below_idx_b;
            }
            // в случаях выше эти индексы лучше вообще не трогать, поэтому не объединяю
            // смотрим, что в ячейке выше диагонали, а именно, как в неё пришли.

            return a[above_idx_a] > b[below_idx_b] ? above_idx_a : sizeA + below_idx_b;
        }

        if (below) {
            l = m;
        } else {
            r = m;
        }
    }
}


// тут нам нужны оба указателия, поэтому вернем примерно sizeB * above_idx_a + above_idx_b
inline uint
merge_pointer_global(__global const uint *a, __local const uint *b, __global uint *c, uint sizeA, uint sizeB, uint diag_index) {
    unsigned int sizeB_inc = sizeB + 1;
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
                return sizeB_inc * above_idx_a + above_idx_b;
            }
            if ((diag_index < sizeB) && m == diag_length - 1) {
                return sizeB_inc * below_idx_a + below_idx_b;
            }
            // в случаях выше эти индексы лучше вообще не трогать, поэтому не объединяю
            return a[above_idx_a] > b[below_idx_b] ? above_idx_a * sizeB_inc + above_idx_b : below_idx_a * sizeB_inc + below_idx_b;
        }

        if (below) {
            l = m;
        } else {
            r = m;
        }
    }
}

inline void
merge_local(__local const uint *a, __local const uint *b, __local uint *c, uint sizeA, uint sizeB) {
    uint diag_index = get_local_id(0);
    if (diag_index >= sizeA + sizeB) return;
    uint mptr = merge_pointer(a, b, c, sizeA, sizeB, diag_index);

    c[diag_index] = mptr > sizeA ? b[mptr - sizeA] : a[mptr];
}

inline void
merge_global(__global const uint *a, __local const uint *b, __global uint *c, uint sizeA, uint sizeB) {
    uint sizeB_inc = sizeB + 1;
    uint step_length = ((sizeA + sizeB) + GROUP_SIZE - 1) / GROUP_SIZE;
    uint diag_index = get_local_id(0) * step_length;
    if (diag_index >= sizeA + sizeB) return;
    uint mptr = merge_pointer_global(a, b, c, sizeA, sizeB, diag_index);
    uint a_ptr = mptr / sizeB_inc;
    uint b_ptr = mptr % sizeB_inc;

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


// Пусть результат валяется в глобальной памяти, копировать туда наш массив необязетельно
inline void
merge_global_local(__global const uint *a, __local const uint *b, __global uint *c, uint sizeA, uint sizeB) {
    // в глобальном случае придется делать это со страйдом
    unsigned int res_size = sizeA + sizeB;
    uint steps = (res_size + GROUP_SIZE - 1) / GROUP_SIZE;
    unsigned int min_side = sizeA < sizeB ? sizeA : sizeB;
    unsigned int max_side = res_size - min_side;

    for (uint step = 0; step < steps; ++step) {
        unsigned int diag_index = get_local_id(0) + GROUP_SIZE * step;
        // we can allow it because there are no barriers in the code below
        if (diag_index >= res_size) return;

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
                    c[diag_index] = a[above_idx_a];
                    return;
                }
                if ((diag_index < sizeB) && m == diag_length - 1) {
                    c[diag_index] = b[below_idx_b];
                    return;
                }
                // в случаях выше эти индексы лучше вообще не трогать, поэтому не объединяю
                bool is_greater = a[above_idx_a] > b[below_idx_b];

                c[diag_index] = is_greater ? a[above_idx_a] : b[below_idx_b];
                return;
            }

            if (below) {
                l = m;
            } else {
                r = m;
            }
        }
    }
}

inline uint
ceil_to_power2(uint v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

inline void
swap_buffers(__local uint *a, __local uint *b) {
    __local uint *tmp = a;
    a = b;
    b = tmp;
}

inline void
swap_buffers_global(__global uint *a, __global uint *b) {
    __global uint *tmp = a;
    a = b;
    b = tmp;
}


__kernel void local_global_merge(__global const uint *indices,
                                 uint group_start, // indices_pointers[workload_group_id]

                                 __global uint *aux_mem_pointers,
                                 __global uint *aux_mem,

                                 __global const uint *pre_matrix_rows_pointers,
                                 __global uint *pre_matrix_cols_indices,
                                 __global uint *nnz_estimation,

                                 __global const uint *a_rows_pointers,
                                 __global const uint *a_cols,

                                 __global const uint *b_rows_pointers,
                                 __global const uint *b_rows_compressed,
                                 __global const uint *b_cols,
                                 const uint b_nzr
) {
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);

    uint col_index, b_row_pointer, b_start, b_end, b_row_length, scan_size, new_length;

    // не получится сливать массивы inplace
    __local uint merge_buffer1[BUFFER_SIZE];
    __local uint merge_buffer2[BUFFER_SIZE];

    // буферы придется свапать, для этого указатели.
    __local uint *buff_1 = merge_buffer1;
    __local uint *buff_2 = merge_buffer2;

    // скан для определения места элемента. +1 так как это exclusive scan, и в самый последний элемент мы запишем сумму
    __local uint positions[BUFFER_SIZE + 1];

    uint fill_pointer = 0;

    uint row_index = indices[group_start + group_id];
    uint a_start = a_rows_pointers[row_index];
    uint a_end = a_rows_pointers[row_index + 1];

    // установим глобальные указатели для удобства
    __global uint *result = pre_matrix_cols_indices + pre_matrix_rows_pointers[row_index];
    __global uint *current_row_aux_memory = aux_mem + aux_mem_pointers[group_id];

    // если дойдет до глобальной памяти
    __global uint *buff_1_global = result;
    __global uint *buff_2_global = current_row_aux_memory;


    __local bool global_flag = false;
    __local uint new_b_row_start = 0;
    __local uint old_b_row_end = 0;
    __local uint global_fill_pointer = 0;

    for (uint a_row_pointer = a_start; a_row_pointer < a_end; ++a_row_pointer) {

        if (!global_flag) {
            col_index = a_cols[a_row_pointer];
            b_row_pointer = search_global(b_rows_compressed, col_index, b_nzr);
            if (b_row_pointer == b_nzr) continue;
        }

        b_start = global_flag ? new_b_row_start : b_rows_pointers[b_row_pointer];
        b_end =  global_flag ? old_b_row_end : b_rows_pointers[b_row_pointer + 1];
        b_row_length = b_end - b_start;
        if (global_flag) {
            global_flag = false;
        }
        // сколько понадобится шагов со страйдом в group_size, их будет не более двух! Так как выделим половину

        uint steps = (b_row_length + GROUP_SIZE - 1) / GROUP_SIZE;

        for (uint group_step = 0; group_step < steps && (!global_flag); ++group_step) {

            uint elem_id_local = group_step * GROUP_SIZE + local_id;
            uint elem_id = b_start + elem_id_local;

            if (elem_id < b_end) {
                uint fill_position = elem_id_local + fill_pointer;
                // fill_position нам тут нужен для проверки, что ряд поместится.
                if (fill_position < BUFFER_SIZE) {
                    // самый первый ряд просто копируем в начальный буфер
                    if (fill_pointer == 0) {
                        buff_2[elem_id_local] = b_cols[elem_id];
                    } else {
                        buff_1[elem_id_local] = b_cols[elem_id];
                        // теперь надо бинпоиском проверить, нет ли этого элемента среди уже имеющихся
                        positions[elem_id_local] =
                                search_local(buff_2, buff_1[elem_id_local], fill_pointer) == fill_pointer ? 1 : 0;
                    }
                } else {
                    // не помещаемся в буфер, возможно (!), придется скидывать в глобальную память
                    a_row_pointer--;
                    if (fill_position == BUFFER_SIZE) {
                        global_flag = true;
                        new_b_row_start = elem_id;
                        old_b_row_end = b_end;
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // посчитать преф суммы на positions, сначала обнулим все что выходим за пределы
        uint filled_b_length = global_flag ? new_b_row_start - b_start : b_row_length;
        if (fill_pointer != 0) {
            if (local_id >= filled_b_length) positions[local_id] = 0;
            scan_size = min(ceil_to_power2(filled_b_length), BUFFER_SIZE);
            scan(positions, scan_size);
            new_length = positions[scan_size];
            // buff_1 --  в начало этого буфера записали элементы
            // buff_2 + fill_pointer -- сюда сейчас скопируем
            barrier(CLK_LOCAL_MEM_FENCE);
            set_positions(positions, buff_1, filled_b_length, buff_2 + fill_pointer, scan_size);
            barrier(CLK_LOCAL_MEM_FENCE);
            // теперь задача смержить две отсортированных половинки из buff_to_merge в buff_1
            merge_local(buff_2, buff_2 + fill_pointer, buff_1, fill_pointer, new_length);
            ///
            swap_buffers(buff_1, buff_2);
        } else {
            new_length = filled_b_length;
        }

        fill_pointer += new_length;

        barrier(CLK_LOCAL_MEM_FENCE);
        // Если кто-то утсановил глобальный флаг и после удалений дубликатов буфер полностью заполнен
        // будем его освобождать

        bool no_place_in_buffer  = fill_pointer == BUFFER_SIZE;
        bool last_step = (a_row_pointer == a_end - 1) && (filled_b_length == b_row_length);

        if (no_place_in_buffer || last_step) {
            // если ещё не копировали в глобальную память, то там не с чем сливать
            if (global_fill_pointer == 0) {
                if (local_id < fill_pointer) {result[local_id] = buff_2[local_id];}
            } else {

                positions[local_id] =
                        search_global(result, buff_2[local_id], global_fill_pointer) == global_fill_pointer ? 1 : 0;
                if (local_id >= fill_pointer) positions[local_id] = 0;

                barrier(CLK_GLOBAL_MEM_FENCE);
                scan_size = min(ceil_to_power2(fill_pointer), BUFFER_SIZE);
                scan(positions, fill_pointer);
                barrier(CLK_LOCAL_MEM_FENCE);
                new_length = positions[scan_size];
                // переместим их из buff2 в buff1
                set_positions(positions, buff_2, fill_pointer, buff_1, scan_size);
                barrier(CLK_LOCAL_MEM_FENCE);

                merge_global_local(buff_1_global, buff_1, buff_2_global, global_fill_pointer, new_length);
                barrier(CLK_LOCAL_MEM_FENCE);
                barrier(CLK_GLOBAL_MEM_FENCE);
                swap_buffers_global(buff_1_global, buff_2_global);
            }
            global_fill_pointer += new_length;
        }
        fill_pointer = 0;
    }

    if (buff_1_global != result && local_id < fill_pointer) {
        result[local_id] = buff_1[local_id];
    }

    nnz_estimation[row_index] = global_fill_pointer;
}