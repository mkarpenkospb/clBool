//#ifndef RUN
//
//#include "../clion_defines.cl"
//#define GROUP_SIZE 256
//
//#endif

#define GROUP_SIZE 256

__kernel void aplusb(__global float* restrict a,
                     __global float* restrict b,
                     __global float* restrict c,
                     unsigned int n)
{
    const unsigned int index = get_global_id(0);
    c[index] = a[index] + b[index];
}

