#ifndef RUN

#include "../clion_defines.cl"
//#define GROUP_SIZE 256
#define __attribute__( a )
#endif
#define GROUP_SIZE 256
#define __local local

#define NNZ_ESTIMATION_4 4
#define NNZ_ESTIMATION_8 8
#define NNZ_ESTIMATION_16 16
#define NNZ_ESTIMATION_20 20
#define NNZ_ESTIMATION_24 24
#define NNZ_ESTIMATION_28 28
#define NNZ_ESTIMATION_32 32

uint search_global(__global const uint *array, uint value, uint size) {
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


void swap(__local uint *array, uint i, uint j) {
    uint tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
}

void swap_unique(__local uint *array, __global uint *result, uint i, uint j, uint k) {
    uint tmp = array[i];
    array[i] = array[j];
    result[k] = tmp;
}

void heapify(__local uint *heap, uint i, uint heap_size) {
    uint current = i;
    uint smallest = current;
    while (true) {
        uint left = 2 * smallest + 1;
        uint right = 2 * smallest + 2;

        if (left < heap_size && heap[smallest] > heap[left]) {
            smallest = left;
        }

        if (right < heap_size && heap[smallest] > heap[right]) {
            smallest = right;
        }

        if (smallest != current) {
            swap(heap, current, smallest);
            current = smallest;
        } else {
            return;
        }
    }
}

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void heap_merge_4(__global const uint* restrict indices,
                         uint group_start, // indices_pointers[workload_group_id]
                         uint group_length,

                         __global const uint* restrict pre_matrix_rows_pointers,
                         __global uint* restrict pre_matrix_cols_indices,

                         __global uint* restrict nnz_estimation,

                         __global const uint* restrict a_rows_pointers,
                         __global const uint* restrict a_cols,

                         __global const uint* restrict b_rows_pointers,
                         __global const uint* restrict b_rows_compressed,
                         __global const uint* restrict b_cols,
                         const uint b_nzr
) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint row_pos = group_start + global_id;
    uint group_end = group_start + group_length;

    if (row_pos >= group_end) return;

    uint a_row_index = indices[row_pos];

    __local uint heap_storage[GROUP_SIZE][NNZ_ESTIMATION_4];
    __local uint *heap = heap_storage[local_id];

    __global uint *result = pre_matrix_cols_indices + pre_matrix_rows_pointers[a_row_index];

    // ------------------ fill heap -------------------

    uint a_start = a_rows_pointers[a_row_index];
    uint a_end = a_rows_pointers[a_row_index + 1];
    uint heap_fill_pointer = 0;

    for (uint a_pointer = a_start; a_pointer < a_end; ++a_pointer) {
        uint col_index = a_cols[a_pointer];
        uint b_row_index = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_row_index == b_nzr) continue;

        uint b_start = b_rows_pointers[b_row_index];
        uint b_end = b_rows_pointers[b_row_index + 1];

        for (uint b_pointer = b_start; b_pointer < b_end; ++b_pointer) {
            heap[heap_fill_pointer] = b_cols[b_pointer];
            ++heap_fill_pointer;
        }
    }


    // ---------------------- heapsort (Min Heap)------------------------------

    /*
     * сделаем min heap, чтобы можно было выложить её в возрастающем порядке сразу в глобальную память,
     * отрывая корни
     */

    uint heap_pointer_unique = 0;
    uint heap_size = NNZ_ESTIMATION_4;

    // build heap
    for (uint i = (NNZ_ESTIMATION_4 / 2); i > 0; --i) {
        heapify(heap, i - 1, heap_size);
    }

    // first step separately
    uint last = heap[0];
    swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);

    ++heap_pointer_unique;
    --heap_size;
    heapify(heap, 0, heap_size);
    // sorting
    for(uint i = 0; i < NNZ_ESTIMATION_4 - 1; ++i) {
        if (heap[0] != last) {
            /*
             * записываем в result следующий корень из кучи только если он не равен тому,
             * что мы туда записали в предыдущий раз
             */
            last = heap[0];
            swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);
            ++heap_pointer_unique;
        } else {
            swap(heap, 0, heap_size - 1);
        }
        --heap_size;
        heapify(heap, 0, heap_size);
    }

    nnz_estimation[a_row_index] = heap_pointer_unique;
}

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void heap_merge_8(__global const uint* restrict indices,
                          uint group_start, // indices_pointers[workload_group_id]
                          uint group_length,

                          __global const uint* restrict pre_matrix_rows_pointers,
                          __global uint* restrict pre_matrix_cols_indices,

                          __global uint* restrict nnz_estimation,

                          __global const uint* restrict a_rows_pointers,
                          __global const uint* restrict a_cols,

                          __global const uint* restrict b_rows_pointers,
                          __global const uint* restrict b_rows_compressed,
                          __global const uint* restrict b_cols,
                          const uint b_nzr
) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint row_pos = group_start + global_id;
    uint group_end = group_start + group_length;

    if (row_pos >= group_end) return;

    uint a_row_index = indices[row_pos];

    __local uint heap_storage[GROUP_SIZE][NNZ_ESTIMATION_8];
    __local uint *heap = heap_storage[local_id];

    __global uint *result = pre_matrix_cols_indices + pre_matrix_rows_pointers[a_row_index];

    // ------------------ fill heap -------------------

    uint a_start = a_rows_pointers[a_row_index];
    uint a_end = a_rows_pointers[a_row_index + 1];
    uint heap_fill_pointer = 0;

    for (uint a_pointer = a_start; a_pointer < a_end; ++a_pointer) {
        uint col_index = a_cols[a_pointer];
        uint b_row_index = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_row_index == b_nzr) continue;

        uint b_start = b_rows_pointers[b_row_index];
        uint b_end = b_rows_pointers[b_row_index + 1];

        for (uint b_pointer = b_start; b_pointer < b_end; ++b_pointer) {
            heap[heap_fill_pointer] = b_cols[b_pointer];
            ++heap_fill_pointer;
        }
    }


    // ---------------------- heapsort (Min Heap)------------------------------

    /*
     * сделаем min heap, чтобы можно было выложить её в возрастающем порядке сразу в глобальную память,
     * отрывая корни
     */

    uint heap_pointer_unique = 0;
    uint heap_size = NNZ_ESTIMATION_8;

    // build heap
    for (uint i = (NNZ_ESTIMATION_8 / 2); i > 0; --i) {
        heapify(heap, i - 1, heap_size);
    }

    // first step separately
    uint last = heap[0];
    swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);

    ++heap_pointer_unique;
    --heap_size;
    heapify(heap, 0, heap_size);
    // sorting
    for(uint i = 0; i < NNZ_ESTIMATION_8 - 1; ++i) {
        if (heap[0] != last) {
            /*
             * записываем в result следующий корень из кучи только если он не равен тому,
             * что мы туда записали в предыдущий раз
             */
            last = heap[0];
            swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);
            ++heap_pointer_unique;
        } else {
            swap(heap, 0, heap_size - 1);
        }
        --heap_size;
        heapify(heap, 0, heap_size);
    }

    nnz_estimation[a_row_index] = heap_pointer_unique;
}

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void heap_merge_16(__global const uint* restrict indices,
                           uint group_start, // indices_pointers[workload_group_id]
                           uint group_length,

                           __global const uint* restrict pre_matrix_rows_pointers,
                           __global uint* restrict pre_matrix_cols_indices,

                           __global uint* restrict nnz_estimation,

                           __global const uint* restrict a_rows_pointers,
                           __global const uint* restrict a_cols,

                           __global const uint* restrict b_rows_pointers,
                           __global const uint* restrict b_rows_compressed,
                           __global const uint* restrict b_cols,
                           const uint b_nzr
) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint row_pos = group_start + global_id;
    uint group_end = group_start + group_length;

    if (row_pos >= group_end) return;

    uint a_row_index = indices[row_pos];

    __local uint heap_storage[GROUP_SIZE][NNZ_ESTIMATION_16];
    __local uint *heap = heap_storage[local_id];

    __global uint *result = pre_matrix_cols_indices + pre_matrix_rows_pointers[a_row_index];

    // ------------------ fill heap -------------------

    uint a_start = a_rows_pointers[a_row_index];
    uint a_end = a_rows_pointers[a_row_index + 1];
    uint heap_fill_pointer = 0;

    for (uint a_pointer = a_start; a_pointer < a_end; ++a_pointer) {
        uint col_index = a_cols[a_pointer];
        uint b_row_index = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_row_index == b_nzr) continue;

        uint b_start = b_rows_pointers[b_row_index];
        uint b_end = b_rows_pointers[b_row_index + 1];

        for (uint b_pointer = b_start; b_pointer < b_end; ++b_pointer) {
            heap[heap_fill_pointer] = b_cols[b_pointer];
            ++heap_fill_pointer;
        }
    }


    // ---------------------- heapsort (Min Heap)------------------------------

    /*
     * сделаем min heap, чтобы можно было выложить её в возрастающем порядке сразу в глобальную память,
     * отрывая корни
     */

    uint heap_pointer_unique = 0;
    uint heap_size = NNZ_ESTIMATION_16;

    // build heap
    for (uint i = (NNZ_ESTIMATION_16 / 2); i > 0; --i) {
        heapify(heap, i - 1, heap_size);
    }

    // first step separately
    uint last = heap[0];
    swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);

    ++heap_pointer_unique;
    --heap_size;
    heapify(heap, 0, heap_size);
    // sorting
    for(uint i = 0; i < NNZ_ESTIMATION_16 - 1; ++i) {
        if (heap[0] != last) {
            /*
             * записываем в result следующий корень из кучи только если он не равен тому,
             * что мы туда записали в предыдущий раз
             */
            last = heap[0];
            swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);
            ++heap_pointer_unique;
        } else {
            swap(heap, 0, heap_size - 1);
        }
        --heap_size;
        heapify(heap, 0, heap_size);
    }

    nnz_estimation[a_row_index] = heap_pointer_unique;
}

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void heap_merge_20(__global const uint* restrict indices,
                            uint group_start, // indices_pointers[workload_group_id]
                            uint group_length,

                            __global const uint* restrict pre_matrix_rows_pointers,
                            __global uint* restrict pre_matrix_cols_indices,

                            __global uint* restrict nnz_estimation,

                            __global const uint* restrict a_rows_pointers,
                            __global const uint* restrict a_cols,

                            __global const uint* restrict b_rows_pointers,
                            __global const uint* restrict b_rows_compressed,
                            __global const uint* restrict b_cols,
                            const uint b_nzr
) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint row_pos = group_start + global_id;
    uint group_end = group_start + group_length;

    if (row_pos >= group_end) return;

    uint a_row_index = indices[row_pos];

    __local uint heap_storage[GROUP_SIZE][NNZ_ESTIMATION_20];
    __local uint *heap = heap_storage[local_id];

    __global uint *result = pre_matrix_cols_indices + pre_matrix_rows_pointers[a_row_index];

    // ------------------ fill heap -------------------

    uint a_start = a_rows_pointers[a_row_index];
    uint a_end = a_rows_pointers[a_row_index + 1];
    uint heap_fill_pointer = 0;

    for (uint a_pointer = a_start; a_pointer < a_end; ++a_pointer) {
        uint col_index = a_cols[a_pointer];
        uint b_row_index = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_row_index == b_nzr) continue;

        uint b_start = b_rows_pointers[b_row_index];
        uint b_end = b_rows_pointers[b_row_index + 1];

        for (uint b_pointer = b_start; b_pointer < b_end; ++b_pointer) {
            heap[heap_fill_pointer] = b_cols[b_pointer];
            ++heap_fill_pointer;
        }
    }


    // ---------------------- heapsort (Min Heap)------------------------------

    /*
     * сделаем min heap, чтобы можно было выложить её в возрастающем порядке сразу в глобальную память,
     * отрывая корни
     */

    uint heap_pointer_unique = 0;
    uint heap_size = NNZ_ESTIMATION_20;

    // build heap
    for (uint i = (NNZ_ESTIMATION_20 / 2); i > 0; --i) {
        heapify(heap, i - 1, heap_size);
    }

    // first step separately
    uint last = heap[0];
    swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);

    ++heap_pointer_unique;
    --heap_size;
    heapify(heap, 0, heap_size);
    // sorting
    for(uint i = 0; i < NNZ_ESTIMATION_20 - 1; ++i) {
        if (heap[0] != last) {
            /*
             * записываем в result следующий корень из кучи только если он не равен тому,
             * что мы туда записали в предыдущий раз
             */
            last = heap[0];
            swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);
            ++heap_pointer_unique;
        } else {
            swap(heap, 0, heap_size - 1);
        }
        --heap_size;
        heapify(heap, 0, heap_size);
    }

    nnz_estimation[a_row_index] = heap_pointer_unique;
}

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void heap_merge_24(__global const uint* restrict indices,
                            uint group_start, // indices_pointers[workload_group_id]
                            uint group_length,

                            __global const uint* restrict pre_matrix_rows_pointers,
                            __global uint* restrict pre_matrix_cols_indices,

                            __global uint* restrict nnz_estimation,

                            __global const uint* restrict a_rows_pointers,
                            __global const uint* restrict a_cols,

                            __global const uint* restrict b_rows_pointers,
                            __global const uint* restrict b_rows_compressed,
                            __global const uint* restrict b_cols,
                            const uint b_nzr
) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint row_pos = group_start + global_id;
    uint group_end = group_start + group_length;

    if (row_pos >= group_end) return;

    uint a_row_index = indices[row_pos];

    __local uint heap_storage[GROUP_SIZE][NNZ_ESTIMATION_24];
    __local uint *heap = heap_storage[local_id];

    __global uint *result = pre_matrix_cols_indices + pre_matrix_rows_pointers[a_row_index];

    // ------------------ fill heap -------------------

    uint a_start = a_rows_pointers[a_row_index];
    uint a_end = a_rows_pointers[a_row_index + 1];
    uint heap_fill_pointer = 0;

    for (uint a_pointer = a_start; a_pointer < a_end; ++a_pointer) {
        uint col_index = a_cols[a_pointer];
        uint b_row_index = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_row_index == b_nzr) continue;

        uint b_start = b_rows_pointers[b_row_index];
        uint b_end = b_rows_pointers[b_row_index + 1];

        for (uint b_pointer = b_start; b_pointer < b_end; ++b_pointer) {
            heap[heap_fill_pointer] = b_cols[b_pointer];
            ++heap_fill_pointer;
        }
    }


    // ---------------------- heapsort (Min Heap)------------------------------

    /*
     * сделаем min heap, чтобы можно было выложить её в возрастающем порядке сразу в глобальную память,
     * отрывая корни
     */

    uint heap_pointer_unique = 0;
    uint heap_size = NNZ_ESTIMATION_24;

    // build heap
    for (uint i = (NNZ_ESTIMATION_24 / 2); i > 0; --i) {
        heapify(heap, i - 1, heap_size);
    }

    // first step separately
    uint last = heap[0];
    swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);

    ++heap_pointer_unique;
    --heap_size;
    heapify(heap, 0, heap_size);
    // sorting
    for(uint i = 0; i < NNZ_ESTIMATION_24 - 1; ++i) {
        if (heap[0] != last) {
            /*
             * записываем в result следующий корень из кучи только если он не равен тому,
             * что мы туда записали в предыдущий раз
             */
            last = heap[0];
            swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);
            ++heap_pointer_unique;
        } else {
            swap(heap, 0, heap_size - 1);
        }
        --heap_size;
        heapify(heap, 0, heap_size);
    }

    nnz_estimation[a_row_index] = heap_pointer_unique;
}

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void heap_merge_28(__global const uint* restrict indices,
                            uint group_start, // indices_pointers[workload_group_id]
                            uint group_length,

                            __global const uint* restrict pre_matrix_rows_pointers,
                            __global uint* restrict pre_matrix_cols_indices,

                            __global uint* restrict nnz_estimation,

                            __global const uint* restrict a_rows_pointers,
                            __global const uint* restrict a_cols,

                            __global const uint* restrict b_rows_pointers,
                            __global const uint* restrict b_rows_compressed,
                            __global const uint* restrict b_cols,
                            const uint b_nzr
) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint row_pos = group_start + global_id;
    uint group_end = group_start + group_length;

    if (row_pos >= group_end) return;

    uint a_row_index = indices[row_pos];

    __local uint heap_storage[GROUP_SIZE][NNZ_ESTIMATION_28];
    __local uint *heap = heap_storage[local_id];

    __global uint *result = pre_matrix_cols_indices + pre_matrix_rows_pointers[a_row_index];

    // ------------------ fill heap -------------------

    uint a_start = a_rows_pointers[a_row_index];
    uint a_end = a_rows_pointers[a_row_index + 1];
    uint heap_fill_pointer = 0;

    for (uint a_pointer = a_start; a_pointer < a_end; ++a_pointer) {
        uint col_index = a_cols[a_pointer];
        uint b_row_index = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_row_index == b_nzr) continue;

        uint b_start = b_rows_pointers[b_row_index];
        uint b_end = b_rows_pointers[b_row_index + 1];

        for (uint b_pointer = b_start; b_pointer < b_end; ++b_pointer) {
            heap[heap_fill_pointer] = b_cols[b_pointer];
            ++heap_fill_pointer;
        }
    }


    // ---------------------- heapsort (Min Heap)------------------------------

    /*
     * сделаем min heap, чтобы можно было выложить её в возрастающем порядке сразу в глобальную память,
     * отрывая корни
     */

    uint heap_pointer_unique = 0;
    uint heap_size = NNZ_ESTIMATION_28;

    // build heap
    for (uint i = (NNZ_ESTIMATION_28 / 2); i > 0; --i) {
        heapify(heap, i - 1, heap_size);
    }

    // first step separately
    uint last = heap[0];
    swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);

    ++heap_pointer_unique;
    --heap_size;
    heapify(heap, 0, heap_size);
    // sorting
    for(uint i = 0; i < NNZ_ESTIMATION_28 - 1; ++i) {
        if (heap[0] != last) {
            /*
             * записываем в result следующий корень из кучи только если он не равен тому,
             * что мы туда записали в предыдущий раз
             */
            last = heap[0];
            swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);
            ++heap_pointer_unique;
        } else {
            swap(heap, 0, heap_size - 1);
        }
        --heap_size;
        heapify(heap, 0, heap_size);
    }

    nnz_estimation[a_row_index] = heap_pointer_unique;
}

__attribute__((reqd_work_group_size(32,1,1)))
__kernel void heap_merge_32(__global const uint* restrict indices,
                            uint group_start, // indices_pointers[workload_group_id]
                            uint group_length,

                            __global const uint* restrict pre_matrix_rows_pointers,
                            __global uint* restrict pre_matrix_cols_indices,

                            __global uint* restrict nnz_estimation,

                            __global const uint* restrict a_rows_pointers,
                            __global const uint* restrict a_cols,

                            __global const uint* restrict b_rows_pointers,
                            __global const uint* restrict b_rows_compressed,
                            __global const uint* restrict b_cols,
                            const uint b_nzr
) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint row_pos = group_start + global_id;
    uint group_end = group_start + group_length;

    if (row_pos >= group_end) return;

    uint a_row_index = indices[row_pos];

    __local uint heap_storage[GROUP_SIZE][NNZ_ESTIMATION_32];
    __local uint *heap = heap_storage[local_id];

    __global uint *result = pre_matrix_cols_indices + pre_matrix_rows_pointers[a_row_index];

    // ------------------ fill heap -------------------

    uint a_start = a_rows_pointers[a_row_index];
    uint a_end = a_rows_pointers[a_row_index + 1];
    uint heap_fill_pointer = 0;

    for (uint a_pointer = a_start; a_pointer < a_end; ++a_pointer) {
        uint col_index = a_cols[a_pointer];
        uint b_row_index = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_row_index == b_nzr) continue;

        uint b_start = b_rows_pointers[b_row_index];
        uint b_end = b_rows_pointers[b_row_index + 1];

        for (uint b_pointer = b_start; b_pointer < b_end; ++b_pointer) {
            heap[heap_fill_pointer] = b_cols[b_pointer];
            ++heap_fill_pointer;
        }
    }


    // ---------------------- heapsort (Min Heap)------------------------------

    /*
     * сделаем min heap, чтобы можно было выложить её в возрастающем порядке сразу в глобальную память,
     * отрывая корни
     */

    uint heap_pointer_unique = 0;
    uint heap_size = NNZ_ESTIMATION_32;

    // build heap
    for (uint i = (NNZ_ESTIMATION_32 / 2); i > 0; --i) {
        heapify(heap, i - 1, heap_size);
    }

    // first step separately
    uint last = heap[0];
    swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);

    ++heap_pointer_unique;
    --heap_size;
    heapify(heap, 0, heap_size);
    // sorting
    for(uint i = 0; i < NNZ_ESTIMATION_32 - 1; ++i) {
        if (heap[0] != last) {
            /*
             * записываем в result следующий корень из кучи только если он не равен тому,
             * что мы туда записали в предыдущий раз
             */
            last = heap[0];
            swap_unique(heap, result, 0, heap_size - 1, heap_pointer_unique);
            ++heap_pointer_unique;
        } else {
            swap(heap, 0, heap_size - 1);
        }
        --heap_size;
        heapify(heap, 0, heap_size);
    }

    nnz_estimation[a_row_index] = heap_pointer_unique;
}