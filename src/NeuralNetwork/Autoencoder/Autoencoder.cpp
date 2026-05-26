#include "Autoencoder.hpp"

Autoencoder::Autoencoder(size_t inputSize, size_t hiddenSize)
{
    layers.reserve(2);
    layers.emplace_back(hiddenSize, hiddenSize, inputSize, Activation::ReLU);
    layers.emplace_back(inputSize, inputSize, hiddenSize, Activation::Sigmoid);
    // Layer constructor already applies Xavier uniform initialization — no need to overwrite
}

Eigen::VectorXf Autoencoder::encode(const Eigen::VectorXf& input) {
    return layers[0].forward(input);
}

Eigen::VectorXf Autoencoder::decode(const Eigen::VectorXf& encoded) {
    return layers[1].forward(encoded);
}

Eigen::VectorXf Autoencoder::forward(const Eigen::VectorXf& x) {
    return decode(encode(x));
}

void Autoencoder::backward(Eigen::VectorXf& delta) {
    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
        if (i == static_cast<int>(layers.size()) - 1) {
            delta = layers[i].backwardOutput(delta);  
        } else {
            delta = layers[i].backward(delta);
        }
    }
}

void Autoencoder::updateWeights(float learningRate) {
    for (size_t i = 0; i < layers.size(); i++) {
        layers[i].updateWeights(learningRate);
    }
}