#include "LossFunctions.hpp"

Eigen::VectorXf LossFunctions::apply(LossFunction lossFunction, const Eigen::VectorXf& y_predicted, const Eigen::VectorXf& y_true) {
    switch (lossFunction)
    {
        case LossFunction::MSE:
            return (y_predicted - y_true).array().square().matrix();

        case LossFunction::BinaryCrossEntropy: {
            // Avoid log(0)
            Eigen::VectorXf clippedPrediction = y_predicted.cwiseMax(1e-7).cwiseMin(1 - 1e-7);
            // Cross-entropy BCE
            return y_true.cwiseProduct(clippedPrediction.array().log().matrix()) +
                                      (1 - y_true.array()).matrix().cwiseProduct((1 - clippedPrediction.array()).log().matrix());
        }
    }

    return Eigen::VectorXf::Zero(y_true.size());
}

Eigen::VectorXf LossFunctions::derivative(LossFunction lossFunction, const Eigen::VectorXf& y_predicted, const Eigen::VectorXf& y_true) {
    switch (lossFunction)
    {
        case LossFunction::MSE:
            return (2 * (y_predicted - y_true)).matrix();

        case LossFunction::BinaryCrossEntropy: {
            return (y_predicted - y_true);
        }
    }

    return Eigen::VectorXf::Zero(y_true.size());
}