//#include "../cl_defines.hpp"

#include "../utils.hpp"
#include "../fast_random.h"
#include "../library_classes/matrix_coo.hpp"
#include "../library_classes/matrix_csr.hpp"
#include "../library_classes/controls.hpp"

#include <vector>
#include <iomanip>
#include <algorithm>
#include <iostream>



void testBitonicSort() {
    // check on different ns

    uint32_t max_size = 32 * 1024 * 1024;
    uint32_t test_num = 1;

    uint32_t n = 0;
    std::vector<uint32_t> rows;
    std::vector<uint32_t> cols;

    FastRandom r(42);

    Controls controls = create_controls();

    for (uint32_t test_iter = 0; test_iter < test_num; ++ test_iter) {
        n = std::abs(r.next()) % max_size;
        std::cout << "for n=" << n << std::endl;
        rows.resize(n);
        cols.resize(n);
        fill_random_matrix(rows, cols);
        uint32_t n_rows = *std::max_element(rows.begin(), rows.end());
        uint32_t n_cols = *std::max_element(cols.begin(), cols.end());

        auto matrix = matrix_coo(controls, n_rows, n_cols, n, rows, cols);
//        sort_arrays(rows, cols);
        check_correctness(matrix.get_rows_indexes_cpu(), matrix.get_cols_indexes_cpu());
    }
}


void testAddition() {




}





int main() {
    testBitonicSort();



}


