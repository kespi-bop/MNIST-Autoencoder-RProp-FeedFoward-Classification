#include "Autoencoder.hpp"
#include "ActivationFunctions.hpp"
#include "LossFunctions.hpp"
#include <cmath>

Autoencoder::Autoencoder(const size_t inputSize, const size_t hiddenSize)
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

    return layers[0].forward(input);
}

Eigen::VectorXf Autoencoder::decode(const Eigen::VectorXf& encoded){
    return layers[1].forward(encoded);
}

Eigen::VectorXf Autoencoder::forward(const Eigen::VectorXf& x){
    hidden = encode(x);
    return decode(hidden);
}

void Autoencoder::backward(Eigen::VectorXf& delta){
    for(int i = layers.size() - 1; i >= 0; i--) {
        if(i == layers.size() - 1) {
            delta = layers[i].backward(delta);     
        } else {
            delta = layers[i].backward(delta);
        }
    }
}