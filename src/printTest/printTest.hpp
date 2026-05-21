#ifndef PRINTTEST_HPP
#define PRINTTEST_HPP

#include <Eigen/Dense>

void printASCIIImage(const Eigen::VectorXf& normalizedImg, const Eigen::VectorXf& oneHotLabel, int index);
void printTest(int num_to_print);

#endif
