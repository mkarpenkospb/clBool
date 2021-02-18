//#ifndef RUN
//
//#include "../clion_defines.cl"
//#define GROUP_SIZE 256
//
//#endif

#define GROUP_SIZE 256

__kernel void aplusb(__global const float* a,
                     __global const float* b,
                     __global       float* c,
                     unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    c[index] = a[index] + b[index];
}

