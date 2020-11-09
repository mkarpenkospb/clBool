// закомментить при запуске
//#include "clion_defines.cl"
//#define BLOCK_SIZE 123
//#define CONV_SIZE 3

bool fit(int idx, int n) {
    return idx >= 0 && idx < n;
}

__kernel void convolution(__global float * a, __global float * b, __global float * c, int n)
{
    int m_half = CONV_SIZE / 2;
    // not unsigned, because I want negatives when it's out of borders
    int i_global = get_global_id(1);
    int j_global = get_global_id(0);

    int i_local = get_local_id(1);
    int j_local = get_local_id(0);

    int i_group_id = get_group_id(1);
    int j_group_id = get_group_id(0);

    // maximum size we need
    __local float local_a[BLOCK_SIZE + (CONV_SIZE / 2) * 2][BLOCK_SIZE + (CONV_SIZE / 2) * 2];
    __local float local_b[CONV_SIZE][CONV_SIZE];

    // limits for data size we need

    int bottom = ((int) min(BLOCK_SIZE * (i_group_id + 1), n)) % BLOCK_SIZE;
    int right =  ((int) min(BLOCK_SIZE * (j_group_id + 1), n)) % BLOCK_SIZE;
    bottom = bottom ? bottom : BLOCK_SIZE;
    right = right ? right : BLOCK_SIZE;

    local_a[i_local + m_half][j_local + m_half] = (i_local < n && j_local < n) ?
            a[i_global * n + j_global] : 0;

    if (i_local < CONV_SIZE && j_local < CONV_SIZE) {
        local_b[i_local][j_local] = b[i_local * CONV_SIZE + j_local];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // edges of local matrix without corner
    int read_idx = 0;
    if (i_local < m_half) {
        read_idx = i_global - m_half;
        local_a[i_local][j_local + m_half] = (fit(read_idx, n) && fit(j_global, n)) ?
                a[read_idx * n + j_global] : 0;

        read_idx = i_global + bottom;
        local_a[i_local + bottom + m_half][j_local + m_half] = (fit(read_idx, n) && fit(j_global, n)) ?
                                             a[read_idx * n + j_global] : 0;
    }

    if (j_local < m_half) {
        read_idx = j_global - m_half;
        local_a[i_local + m_half][j_local] = (fit(i_global, n) && fit(read_idx, n)) ?
                a[i_global * n + read_idx] : 0;

        read_idx = j_global + right;
        local_a[i_local + m_half][j_local + right + m_half] = (fit(i_global, n) && fit(read_idx, n)) ?
                a[i_global * n + read_idx] : 0;
    }

    // corners of local matrix
    int i_read = 0;
    int j_read = 0;
    if (i_local < m_half && j_local < m_half) {
        i_read = i_global - m_half;
        j_read = j_global - m_half;
        local_a[i_local][j_local] = fit(i_read, n) && fit(j_read, n) ? a[i_read * n + j_read] : 0;

        i_read = i_global - m_half;
        j_read = j_global + right;
        local_a[i_local][j_local + right + m_half] = fit(i_read, n) && fit(j_read, n) ? a[i_read * n + j_read] : 0;

        i_read = i_global + bottom;
        j_read = j_global - m_half;
        local_a[i_local + bottom + m_half][j_local] = fit(i_read, n) && fit(j_read, n) ? a[i_read * n + j_read] : 0;

        i_read = i_global + bottom;
        j_read = j_global + right;
        local_a[i_local + bottom + m_half][j_local + right + m_half] = fit(i_read, n) && fit(j_read, n) ? a[i_read * n + j_read] : 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float acc = 0;
    for (int i = 0; i < CONV_SIZE; ++i) {
        for (int j = 0; j < CONV_SIZE; ++j) {
            acc += local_b[i][j] * local_a[i_local + i][j_local + j];
        }
    }

    if (i_global < n && j_global < n) {
        c[i_global * n + j_global] = acc;
    }
}