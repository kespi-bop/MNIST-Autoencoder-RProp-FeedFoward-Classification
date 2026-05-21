#pragma once

#include <Eigen/Dense>
#include <vector>
#include "NeuralNetwork.hpp"

class Autoencoder : public NeuralNetwork{
public:
    Autoencoder(const size_t inputSize, const size_t hiddenSize);

    Eigen::VectorXf encode(const Eigen::VectorXf& input);
    Eigen::VectorXf decode(const Eigen::VectorXf& encoded);

    // Forward pass (encode then decode)
    Eigen::VectorXf forward(const Eigen::VectorXf& x) override;

    // Backward pass (gradient computation)
    Eigen::VectorXf backward(const Eigen::VectorXf& lossGradient) override;

    // Update weights using the given learning rate
    void updateWeights(float learningRate);

private:
    const size_t inputSize;
    const size_t hiddenSize;

    // Cache (used during backpropagation)
    Eigen::VectorXf hidden;
};