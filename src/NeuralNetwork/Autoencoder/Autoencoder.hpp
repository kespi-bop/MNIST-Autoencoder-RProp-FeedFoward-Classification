#pragma once

#include <Eigen/Dense>
#include "NeuralNetwork.hpp"

class Autoencoder : public NeuralNetwork {
public:
    Autoencoder(size_t inputSize, size_t hiddenSize);

    Eigen::VectorXf forward(const Eigen::VectorXf& input) override;
    void backward(Eigen::VectorXf& lossGradient) override;
    void updateWeightsRPROP(unsigned int trainingSetSize) override;
    
    const Eigen::VectorXf getEncoding(const Eigen::VectorXf& input) { return encode(input); };

private:
    Eigen::VectorXf encode(const Eigen::VectorXf& input);
    Eigen::VectorXf decode(const Eigen::VectorXf& encoded);
};