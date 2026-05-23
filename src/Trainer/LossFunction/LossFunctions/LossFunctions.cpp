#include "LossFunctions.hpp"

Eigen::VectorXf LossFunctions::apply(LossFunction lossFunction, const Eigen::VectorXf& input, const Eigen::VectorXf& y_predicted, const Eigen::VectorXf& y_true) {
    switch (lossFunction)
    {
        case LossFunction::MSE:
            return (y_predicted - y_true).array().square().matrix();

        case LossFunction::BinaryCrossEntropy: {
            // Avoid log(0)
            Eigen::VectorXf clippedPrediction = y_predicted.cwiseMax(1e-7).cwiseMin(1 - 1e-7);
            // Cross-entropy BCE
            return input.cwiseProduct(clippedPrediction.array().log().matrix()) +
                                      (1 - input.array()).matrix().cwiseProduct((1 - clippedPrediction.array()).log().matrix());
        }
    }

    return Eigen::VectorXf::Zero(input.size());
}

Eigen::VectorXf LossFunctions::derivative(LossFunction lossFunction, const Eigen::VectorXf& input, const Eigen::VectorXf& y_predicted, const Eigen::VectorXf& y_true) {
    switch (lossFunction)
    {
        case LossFunction::MSE:
            return (2 * (y_predicted - y_true)).matrix();

        case LossFunction::BinaryCrossEntropy: {
            // Avoid log(0)
            Eigen::VectorXf clippedPrediction = y_predicted.cwiseMax(1e-7).cwiseMin(1 - 1e-7);
            // Cross-entropy BCE derivative
            return (clippedPrediction - input).array() / (clippedPrediction.array() * (1.0f - clippedPrediction.array()));
        }
    }

    return Eigen::VectorXf::Zero(input.size());
}