//#ifndef RUN
//
//#include "../clion_defines.cl"
//#define GROUP_SIZE 256
//
//#endif

#define GROUP_SIZE 256

__kernel void aplusb(__global uint* restrict a,
                     __global uint* restrict b,
                     __global uint* restrict c,
                     uint n)
{
    const unsigned int index = get_global_id(0);
    c[index] = a[index] + b[index];
}

