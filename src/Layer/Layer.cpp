#include "Layer.hpp"
#include "ActivationFunctions.hpp"

// Constructor with pre-allocation
Layer::Layer(const size_t biasSize, const size_t weightRows, const size_t weightCols, Activation activationFunction) :
    activation(activationFunction)
{
    bias.resize(biasSize);
    weight.resize(weightRows, weightCols);
    
    d_weight.resize(weightRows, weightCols);
    d_bias.resize(biasSize);
}

Eigen::VectorXf Layer::forward(const Eigen::VectorXf& input) {
    lastInput = input;
    a = weight * input + bias;
    z = ActivationFunctions::apply(activation, a);
    return z;
}

Eigen::VectorXf Layer::backward(Eigen::VectorXf& delta) {
    Eigen::VectorXf lossGradient = delta;   

    lossGradient = lossGradient.cwiseProduct(ActivationFunctions::derivative(activation, z));

    d_weight += lossGradient * lastInput.transpose();
    d_bias += lossGradient;
    
    return weight.transpose() * lossGradient;
}
