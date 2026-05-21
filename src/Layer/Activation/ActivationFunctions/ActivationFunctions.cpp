#include "ActivationFunctions.hpp"
#include <cmath>

Eigen::VectorXf ActivationFunctions::apply(Activation act, const Eigen::VectorXf& x)
{
    switch (act)
    {
        case Activation::ReLU:
            return x.cwiseMax(0.0f);

        case Activation::Sigmoid:
            return (1.0f / (1.0f + (-x.array()).exp())).matrix();

        case Activation::Tanh:
            return x.array().tanh().matrix();

        case Activation::Linear:
            return x;

        case Activation::Softmax: {
            Eigen::ArrayXf shifted = x.array() - x.maxCoeff();
            Eigen::ArrayXf expValues = shifted.exp();
            return (expValues / expValues.sum()).matrix();
        }
    }

    return x;
}