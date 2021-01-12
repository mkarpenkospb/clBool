//#include "clion_defines.cl"

//#define SWAP(a,b) {__local uint * tmp=a; a=b; b=tmp;}

//#define GROUP_SIZE 256
// we want to generate code for 31 different heap sizes, and we'll send this
// constant as a compilation parameter
//#define NNZ_ESTIMATION 32

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


void swap(__local uint *array, uint i, uint j) {
    uint tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
}

void swap_unique(__local uint *array, uint i, uint j, uint k) {
    uint tmp = array[i];
    array[i] = array[j];
    array[k] = tmp;
}

void heapify(__local uint *heap, uint i, uint heap_size) {
    uint largest = i;
    while (true) {
        uint left = 2 * largest + 1;
        uint right = 2 * largest + 2;

        if (left < heap_size && heap[i] < heap[left]) {
            largest = left;
        }

        if (right < heap_size && heap[i] < heap[right]) {
            largest = right;
        }

        if (largest != i) {
            swap(heap, i, largest);
        } else {
            return;
        }
    }
}

/*
 * indices -- array of indices to work with
 * group_start, group_length - start and length of the group with nnz-estimation of NNZ_ESTIMATION
 *
 */


/*
 * Размер группы 32 по потоки, иначе too much для локальной памяти
 */

__kernel void heap_merge(__global const unsigned int *indices,
                         unsigned int group_start,
                         unsigned int group_length,

                         __global const unsigned int *a_rows_pointers,
                         __global const unsigned int *a_cols,

                         __global const unsigned int *b_rows_pointers,
                         __global const unsigned int *b_rows_compressed,
                         __global const unsigned int *b_cols,
                         const unsigned int b_nzr
) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint row_pos = group_start + global_id;
    uint group_end = group_start + group_length;
    /*
     * Каждый поток независимо работает со своей пирамидой,
     * поэтому в кернеле нет барьеров, можно сейчас выходить
     */
    if (row_pos >= group_end) return;

    /*
     * a_row_index is not the row pointer itself, but a position where we can find our row pointer in a_rows_pointers
     */

    uint a_row_index = indices[row_pos];

    __local uint heap_storage[GROUP_SIZE][NNZ_ESTIMATION];
    __local uint *heap = heap_storage[local_id];

    // ------------------ fill heap -------------------

    uint a_start = a_rows_pointers[a_row_index];
    uint a_end = a_rows_pointers[a_row_index + 1]; // не выйдем за границы, так как последний указатель отвечает за размер последнего ряда
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


    // ---------------------- heapsort ------------------------------

    uint heap_pointer_unique = NNZ_ESTIMATION;
    uint heap_size = NNZ_ESTIMATION;

    // build heap
    for (uint i = (NNZ_ESTIMATION / 2); i > 0; --i) {
        heapify(heap, i - 1, heap_size);
    }

    // first step separately
    swap(heap, 0, heap_pointer_unique - 1);
    --heap_pointer_unique;
    --heap_size;
    heapify(heap, 0, heap_size);
    // sorting
    for(uint i = 0; i < NNZ_ESTIMATION - 1; ++i) {
        if (heap[0] != heap[heap_pointer_unique]) {
            /*
             * положить в голову следующй элемент из кучи, то есть heap_size - 1,
             * но саму голову на место следующего уникального элемента, то есть heap_pointer_unique - 1
             */
            swap_unique(heap, 0, heap_size - 1, heap_pointer_unique - 1);
            --heap_pointer_unique;
        } else {
            swap(heap, 0, heap_size - 1);
        }
        --heap_size;
        heapify(heap, 0, heap_size);
    }
}