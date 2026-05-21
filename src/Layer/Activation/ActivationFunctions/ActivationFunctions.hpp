#pragma once

#include "Activation.hpp"
#include <Eigen/Dense>

namespace ActivationFunctions {

Eigen::VectorXf apply(Activation act, const Eigen::VectorXf& x);

//Eigen::VectorXf derivative(Activation act, const Eigen::VectorXf& x);

}