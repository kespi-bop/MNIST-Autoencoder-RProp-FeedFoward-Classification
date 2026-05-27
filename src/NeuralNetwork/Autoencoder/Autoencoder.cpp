#include "Autoencoder.hpp"

Autoencoder::Autoencoder(size_t inputSize, size_t hiddenSize) {
    layers.reserve(2);
    layers.emplace_back(hiddenSize, hiddenSize, inputSize,  Activation::ReLU);
    layers.emplace_back(inputSize,  inputSize,  hiddenSize, Activation::Sigmoid);
}

Eigen::VectorXf Autoencoder::encode(const Eigen::VectorXf& input) {
    return layers[0].forward(input);
}

Eigen::VectorXf Autoencoder::decode(const Eigen::VectorXf& encoded) {
    return layers[1].forward(encoded);
}

Eigen::VectorXf Autoencoder::forward(const Eigen::VectorXf& input) {
    return decode(encode(input));
}

void Autoencoder::backward(Eigen::VectorXf& lossGradient) {
    const int lastLayerIndex = static_cast<int>(layers.size()) - 1;

    // Output layer: gradient flows directly (no activation derivative applied)
    lossGradient = layers[lastLayerIndex].backwardOutputLayer(lossGradient);

    // Hidden layers: activation derivative is applied inside backward()
    for (int i = lastLayerIndex - 1; i >= 0; --i)
        lossGradient = layers[i].backward(lossGradient);
}

void Autoencoder::updateWeightsRPROP(unsigned int trainingSetSize) {
    for (auto& layer : layers)
        layer.updateWeightsRPROP(trainingSetSize);
}