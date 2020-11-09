#pragma once
#include <vector>
#include <string>

typedef float value_type;

void convoluion(const std::vector<value_type>& a,
                const std::vector<value_type>& b,
                std::vector<value_type>& c,
                int n,
                int m);

void print_matrix(const std::vector<value_type>& a, int n);

void check_correctness(const std::vector<value_type> &a,
                       const std::vector<value_type> &b,
                       const std::vector<value_type> &c,
                       int n,
                       int m);

void parse_input(const std::string& filename,
                 std::vector<value_type>& a,
                 std::vector<value_type>& b,
                 int& n, int& m);

void write_result(const std::string& filename,
                  const std::vector<value_type>& c,
                  int n);

void fill_random_matrix(std::vector<value_type>& a, int nxn);