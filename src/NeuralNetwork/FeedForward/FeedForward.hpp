#pragma once

#include <Eigen/Dense>
#include "NeuralNetwork.hpp"

class FeedForward : public NeuralNetwork {
public:
    FeedForward(size_t inputSize, size_t hiddenSize, size_t outputSize);

    Eigen::VectorXf forward(const Eigen::VectorXf& input) override;
    void backward(Eigen::VectorXf& lossGradient) override;
    void updateWeightsRPROP(unsigned int trainingSetSize) override;
};