#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#include "convolution.hpp"

#include <vector>
#include <iostream>
#include <iomanip>
#include <iterator>

typedef float value_type;

void test_convolution() {
    int n = 1024;
    int m = 9;
    std::vector<value_type> a(n * n);
    std::vector<value_type> b(m * m);
    std::vector<value_type> c(n * n);
    fill_random_matrix(a, n * n);
    fill_random_matrix(b, m * m);
    convoluion(a, b, c, n, m);
    check_correctness(a, b, c, n, m);
    std::cout << "succeed\n";
}


int main() {
    test_convolution();
//    int n = 0;
//    int m = 0;
//    std::vector<value_type> a;
//    std::vector<value_type> b;
//    std::vector<value_type> c;
//    parse_input("../resources/input2.txt", a, b, n, m);
//    c.resize(n * n);
//    convoluion(a, b, c, n, m);
//    check_correctness(a, b, c, n, m);
////    print_matrix(c, n);
//    write_result("../resources/output2.txt", c, n);
    return 0;
}