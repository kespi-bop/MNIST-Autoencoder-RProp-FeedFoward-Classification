#pragma once

#include <Eigen/Dense>
#include <vector>
#include "NeuralNetwork.hpp"

class Autoencoder : public NeuralNetwork{
public:
    Autoencoder(const size_t inputSize, const size_t hiddenSize);

    // Forward pass (encode then decode)
    Eigen::VectorXf forward(const Eigen::VectorXf& x) override;

    // Backward pass (gradient computation)
    void backward(Eigen::VectorXf& delta) override;

    // Update weights using the given learning rate
    void updateWeights(float learningRate);

private:

    Eigen::VectorXf encode(const Eigen::VectorXf& input);
    Eigen::VectorXf decode(const Eigen::VectorXf& encoded);

    // Cache (used during backpropagation)
    Eigen::VectorXf hidden;
};