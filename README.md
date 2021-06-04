# clBool

**clBool** is a library for operations with sparse boolean matrices, implemented using
the technology OpenCL.

This library is used as a backend in the project [**spbla**](https://github.com/JetBrains-Research/spbla), which primary purpose
is to implement and develop matrix algorithms for context-free (CFPQ) and regular path (RPQ)
queries.

### Operations

 The library provides following operations over a boolean semiring:

 - matrix-matrix multiplication
 - elementwise matrix addition 
 - matrix transpose
 - Kronecker product
 - matrix reduce
 - submatrix

### Matrix formats

There are different formats for operations:

| Operation                             | Format           | 
|---                                                                                                |---            | 
| matrix-matrix multiplication          | DCSR             | 
| elementwise matrix addition           | COO, CSR         | 
| matrix transpose                      | COO              | 
| Kronecker product                     | **DCSR**, COO    | 
| matrix reduce                         | DCSR             |         
| submatrix                             | DCSR             | 

We choose COO and DCSR formats because they need O(nnz) memory space, which can be useful for some hypersparse 
matrices appearing in CFPQ and RPQ, where nnz is a number of nonzero values in a matrix.

**Addition** in COO format is faster on large matrices with almost equally distributed nonzero values per row,
but it demands nearly twice as much memory as CSR version. CSR-based implementation is faster for matrices with 
dense rows.

**For matrix multiplication** there are two algorithms, `matrix_multoplication` and 
`matrix_multiplication_hash`, and the second is preferable to use.

**Kronecker product** is implemented for DCSR and COO formats, and DCSR implementation is mush more faster.

**Conversions** between matrices are also available.

### Simple example

```c++
#include <clbool.hpp>
    int main() {
    // print all available platforms and devices
    clbool::show_devices();
    
    // choose platform id and device id
    clbool::Controls controls = clbool::create_controls(0, 0);

    // ----------------------------- COO ------------------------------------
    uint32_t a_nrows = 5, a_ncols = 5, a_nnz = 6;
    std::vector<uint32_t> a_rows = {0, 0, 0, 2, 2, 4};
    std::vector<uint32_t> a_cols = {0, 1, 4, 2, 3, 2};


    uint32_t b_nrows = 5, b_ncols = 5, b_nnz = 7;
    std::vector<uint32_t> b_rows = {1, 1, 2, 3, 3, 3, 5};
    std::vector<uint32_t> b_cols = {0, 4, 2, 2, 3, 4, 2};

    clbool::matrix_coo a_coo(controls, a_rows.data(), a_cols.data(), a_nrows, a_ncols, a_nnz);
    clbool::matrix_coo b_coo(controls, b_rows.data(), b_cols.data(), b_nrows, b_ncols, b_nnz);

    clbool::matrix_coo c_coo;
    clbool::coo::matrix_addition(controls, c_coo, a_coo, b_coo);
    
    // ----------------------------- DCSR --------------------------------
    clbool::matrix_dcsr a_dcsr = clbool::coo_to_dcsr_shallow(controls, a_coo);
    clbool::matrix_dcsr b_dcsr = clbool::coo_to_dcsr_shallow(controls, b_coo);

    clbool::matrix_dcsr c_dcsr;
    clbool::dcsr::matrix_multiplication_hash(controls, c_dcsr, a_dcsr, b_dcsr);
    clbool::dcsr::kronecker_product(controls, c_dcsr, a_dcsr, b_dcsr);
    clbool::dcsr::reduce(controls, c_dcsr, a_dcsr);
    clbool::dcsr::submatrix(controls, c_dcsr, a_dcsr, 0, 2, 3, 2);

    // ----------------------------- CSR --------------------------------

    clbool::matrix_csr a_csr = clbool::dcsr_to_csr(controls, a_dcsr);
    clbool::matrix_csr b_csr = clbool::dcsr_to_csr(controls, b_dcsr);

    clbool::matrix_csr c_csr;
    clbool::csr::matrix_addition(controls, c_csr, a_csr, b_csr);

    return 0;
}

```


### How to build

Install OpenCL library for your device. 

Get the code and init gtest module 
```shell
git clone https://github.com/mkarpenkospb/clBool
cd clBool
git submodule update --init --recursive
```

For Windows you should define cmake options:  
```shell
 -DOpenCL_LIBRARY=<path-to-opencl>/OpenCL.dll     
 -DOpenCL_INCLUDE_DIR=libs/clew
```

Build library


cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenCL_LIBRARY=C:/Windows/System32/OpenCL.dll -DOpenCL_INCLUDE_DIR=libs/clew -G "MinGW Makefiles"



### How to use




