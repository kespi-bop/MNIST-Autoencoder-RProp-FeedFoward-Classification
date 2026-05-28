#pragma once

#include <Eigen/Dense>

enum class LossFunction {
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy
};