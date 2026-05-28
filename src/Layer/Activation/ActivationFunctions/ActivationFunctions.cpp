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
            float maxVal = x.maxCoeff();
            Eigen::VectorXf expZ = (x.array() - maxVal).exp();
            float sumExp = expZ.sum();
            return expZ / sumExp;
        }
    }

    return Eigen::VectorXf::Zero(x.size());
}

Eigen::VectorXf ActivationFunctions::derivative(Activation act, const Eigen::VectorXf& x)
{
    switch (act)
    {
        case Activation::ReLU:
            return (x.array() > 0).cast<float>();

        case Activation::Sigmoid:
            return x.array() * (1.0f - x.array());

        case Activation::Tanh:
            return 1.0 - x.array().square();

        case Activation::Linear:
            return Eigen::VectorXf::Ones(x.size());

        case Activation::Softmax: {
            // For softmax, the Jacobian is needed, but often we use the diagonal
            // which is y * (1 - y) if we treat it as a single output
            // But for a vector input, the derivative wrt x is more complex.
            // However, in the context of backprop for a single neuron (or if treating
            // as component-wise for simplicity, though incorrect for true softmax jacobian),
            // we approximate.
            // The correct derivative of softmax for a single output is y_i * (1 - y_i)
            // For multiple outputs, it's a matrix.
            // Here we return the diagonal part for simplicity in this vector context,
            // but be aware this is an approximation if treating this as the full derivative.
            Eigen::VectorXf softmax = (x.array() - x.maxCoeff()).exp();
            softmax /= softmax.sum();
            return (softmax.array() * (1.0f - softmax.array())).matrix();
        }
    }

    return Eigen::VectorXf::Zero(x.size());
}