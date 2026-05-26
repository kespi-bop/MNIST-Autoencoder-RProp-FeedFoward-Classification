#pragma once

#include "LossFunction.hpp"
#include <Eigen/Dense>

namespace LossFunctions {

Eigen::VectorXf apply(LossFunction lossFunction, const Eigen::VectorXf& y_predicted, const Eigen::VectorXf& y_true);

Eigen::VectorXf derivative(LossFunction lossFunction, const Eigen::VectorXf& y_predicted, const Eigen::VectorXf& y_true);

}