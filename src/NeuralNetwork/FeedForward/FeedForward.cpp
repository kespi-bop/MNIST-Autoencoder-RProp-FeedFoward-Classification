#include "FeedForward.hpp"

FeedForward::FeedForward(size_t inputSize, size_t hiddenSize, size_t outputSize) {
    layers.reserve(2);
    layers.emplace_back(hiddenSize, hiddenSize, inputSize,  Activation::Sigmoid);
    layers.emplace_back(outputSize, outputSize, hiddenSize, Activation::Softmax);
}

Eigen::VectorXf FeedForward::forward(const Eigen::VectorXf& input) {
    Eigen::VectorXf output = input;
    for(auto& layer : layers)
        output = layer.forward(output);
    return output;
}

void FeedForward::backward(Eigen::VectorXf& lossGradient) {
    const int lastLayerIndex = static_cast<int>(layers.size()) - 1;

    lossGradient = layers[lastLayerIndex].backwardOutputLayer(lossGradient);

    for (int i = lastLayerIndex - 1; i >= 0; --i)
        lossGradient = layers[i].backward(lossGradient);
}

void FeedForward::updateWeightsRPROP(unsigned int trainingSetSize) {
    for (auto& layer : layers)
        layer.updateWeightsRPROP(trainingSetSize);
}