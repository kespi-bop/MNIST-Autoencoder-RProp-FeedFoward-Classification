#include "Layer.hpp"
#include "ActivationFunctions.hpp"
#include <random>
#include <iostream>

Layer::Layer(size_t biasSize, size_t weightRows, size_t weightCols, Activation activationFunction) :
    weightRows(weightRows),
    weightCols(weightCols),
    biasSize(biasSize),
    activation(activationFunction)
{    
    // Xavier uniform initialization
    static std::mt19937 gen(std::random_device{}());
    float bound = std::sqrt(6.0f / (weightRows + weightCols));
    std::uniform_real_distribution<float> dist(-bound, bound);

    weight = Eigen::MatrixXf::NullaryExpr(weightRows, weightCols, [&](){ return dist(gen); });
    bias   = Eigen::VectorXf::NullaryExpr(biasSize,               [&](){ return dist(gen); });

    // RProp state initialization
    const float delta0 = 0.1f;

    d_weight      = Eigen::MatrixXf::Zero(weightRows, weightCols);
    d_bias        = Eigen::VectorXf::Zero(biasSize);

    prev_d_weight = Eigen::MatrixXf::Zero(weightRows, weightCols);
    prev_d_bias   = Eigen::VectorXf::Zero(biasSize);

    delta_weight  = Eigen::MatrixXf::Constant(weightRows, weightCols, delta0);
    delta_bias    = Eigen::VectorXf::Constant(biasSize, delta0);
}

Eigen::VectorXf Layer::forward(const Eigen::VectorXf& input) {
    lastInput = input;
    a = weight * input + bias;
    z = ActivationFunctions::apply(activation, a);
    return z;
}

Eigen::VectorXf Layer::backwardOutput(const Eigen::VectorXf& delta) {
    d_weight += delta * lastInput.transpose();
    d_bias   += delta;
    return weight.transpose() * delta;
}

Eigen::VectorXf Layer::backward(Eigen::VectorXf& delta) {
    Eigen::VectorXf lossGradient = delta.cwiseProduct(ActivationFunctions::derivative(activation, a));

    d_weight += lossGradient * lastInput.transpose();
    d_bias   += lossGradient;
    
    return weight.transpose() * lossGradient;
}

void Layer::updateWeights(float /*learningRate*/) {
    const float etaPlus  = 1.2f;
    const float etaMinus = 0.5f;
    const float deltaMin = 1e-6f;
    const float deltaMax = 50.0f;   

    // --- RProp update for weights ---
    for (int r = 0; r < d_weight.rows(); r++) {
        for (int c = 0; c < d_weight.cols(); c++) {
            float sign_change = prev_d_weight(r,c) * d_weight(r,c);

            if (sign_change > 0.0f) {
                delta_weight(r,c) = std::min(delta_weight(r,c) * etaPlus, deltaMax);
            } else if (sign_change < 0.0f) {
                delta_weight(r,c) = std::max(delta_weight(r,c) * etaMinus, deltaMin);
                weight(r,c) += std::copysign(delta_weight(r,c), prev_d_weight(r,c));
                d_weight(r,c) = 0.0f;
            } 

            weight(r, c) -= std::copysign(delta_weight(r,c), d_weight(r,c));
        }
    }

    // --- RProp update for biases ---
    for (int i = 0; i < d_bias.size(); i++) {
        float sign_change = prev_d_bias(i) * d_bias(i);

        if (sign_change > 0.0f) {
            delta_bias(i) = std::min(delta_bias(i) * etaPlus, deltaMax);
        } else if (sign_change < 0.0f) {
            delta_bias(i) = std::max(delta_bias(i) * etaMinus, deltaMin);
            bias(i) += std::copysign(delta_bias(i), prev_d_bias(i));
            d_bias(i) = 0.0f;
        } 
        bias(i) -= std::copysign(delta_bias(i), d_bias(i));
    }

    prev_d_weight = d_weight.eval();
    prev_d_bias   = d_bias.eval();
    d_weight.setZero();
    d_bias.setZero();
}