#include "Autoencoder.hpp"
#include "ActivationFunctions.hpp"
#include <cmath>

Autoencoder::Autoencoder(const size_t inputSize, const size_t hiddenSize)
    : inputSize(inputSize), hiddenSize(hiddenSize)
{
    layers.reserve(2);
    layers.push_back(Layer(hiddenSize, hiddenSize, inputSize, Activation::ReLU));
    layers.push_back(Layer(inputSize, inputSize, hiddenSize, Activation::Sigmoid));


    // TODO: replace with Xavier initialization
    layers[0].W() = Eigen::MatrixXf::Random(hiddenSize, inputSize);
    layers[0].b() = Eigen::VectorXf::Zero(hiddenSize);

    layers[1].W() = Eigen::MatrixXf::Random(inputSize, hiddenSize);
    layers[1].b() = Eigen::VectorXf::Zero(inputSize);
}

Eigen::VectorXf Autoencoder::encode(const Eigen::VectorXf& input){
    return ActivationFunctions::apply(layers[0].getActivation(), layers[0].W() * input + layers[0].b());
}

Eigen::VectorXf Autoencoder::decode(const Eigen::VectorXf& encoded){
    return ActivationFunctions::apply(layers[1].getActivation(), layers[1].W() * encoded + layers[1].b());
}

Eigen::VectorXf Autoencoder::forward(const Eigen::VectorXf& x){
    hidden = encode(x);
    return decode(hidden);
}

Eigen::VectorXf Autoencoder::backward(const Eigen::VectorXf& lossGradient){
    return lossGradient;
}