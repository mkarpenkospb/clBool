# clBool

## Handle OpenCL

For Windows you should define cmake options:  
 -DOpenCL_LIBRARY=<path-to-opencl>/OpenCL.dll   
 -DOpenCL_INCLUDE_DIR=libs/clew

### Operations

 Library provides the following operations over a boolean semiring:

 - matrix-matrix multiplication
 - elementwise matrix addition 
 - matrix transpose
 - Kronecker product
 - matrix reduce
 - submatrix

