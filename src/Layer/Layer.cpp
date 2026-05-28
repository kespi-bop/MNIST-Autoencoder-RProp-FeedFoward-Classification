#include "Layer.hpp"
#include "ActivationFunctions.hpp"
#include <random>

Layer::Layer(size_t outputSize, size_t weightRows, size_t inputSize, Activation activationFunction)
    : activationType(activationFunction)
{
    static std::mt19937 rng(std::random_device{}());
    const float xavierBound = std::sqrt(6.0f / (weightRows + inputSize));
    std::uniform_real_distribution<float> dist(-xavierBound, xavierBound);

    weights = Eigen::MatrixXf::NullaryExpr(weightRows, inputSize, [&](){ return dist(rng); });
    biases  = Eigen::VectorXf::NullaryExpr(outputSize,             [&](){ return dist(rng); });

    const float initialRpropStep = 0.1f;

    weightGradients     = Eigen::MatrixXf::Zero(weightRows, inputSize);
    biasGradients       = Eigen::VectorXf::Zero(outputSize);
    prevWeightGradients = Eigen::MatrixXf::Zero(weightRows, inputSize);
    prevBiasGradients   = Eigen::VectorXf::Zero(outputSize);
    rpropWeightSteps    = Eigen::MatrixXf::Constant(weightRows, inputSize, initialRpropStep);
    rpropBiasSteps      = Eigen::VectorXf::Constant(outputSize, initialRpropStep);
}

Eigen::VectorXf Layer::forward(const Eigen::VectorXf& input) {
    cachedInput    = input;
    preActivation  = weights * input + biases;
    postActivation = ActivationFunctions::apply(activationType, preActivation);
    return postActivation;
}

// --- Private helpers ---

Eigen::VectorXf Layer::accumulateAndPropagate(const Eigen::VectorXf& gradient) {
    weightGradients += gradient * cachedInput.transpose();
    biasGradients   += gradient;
    return weights.transpose() * gradient;
}

void Layer::applyRpropUpdate(float& param, float& grad, float& prevGrad, float& stepSize) {
    constexpr float etaPlus  = 1.2f;
    constexpr float etaMinus = 0.5f;
    constexpr float deltaMin = 1e-6f;
    constexpr float deltaMax = 50.0f;

    const float signChange = prevGrad * grad;

    if (signChange > 0.0f) {
        stepSize = std::min(stepSize * etaPlus, deltaMax);
    } else if (signChange < 0.0f) {
        stepSize = std::max(stepSize * etaMinus, deltaMin);
        param += std::copysign(stepSize, prevGrad);
        grad = 0.0f;
    }
    param -= std::copysign(stepSize, grad);
}

// --- Public backward pass ---

Eigen::VectorXf Layer::backwardOutputLayer(const Eigen::VectorXf& lossGradient) {
    return accumulateAndPropagate(lossGradient);
}

Eigen::VectorXf Layer::backward(Eigen::VectorXf& incomingGradient) {
    const Eigen::VectorXf localGradient = incomingGradient.cwiseProduct(
        ActivationFunctions::derivative(activationType, postActivation));
    return accumulateAndPropagate(localGradient);
}

void Layer::updateWeightsRPROP(unsigned int trainingSetSize) {
    weightGradients /= static_cast<float>(trainingSetSize);
    biasGradients   /= static_cast<float>(trainingSetSize);

    for (int r = 0; r < weightGradients.rows(); ++r)
        for (int c = 0; c < weightGradients.cols(); ++c)
            applyRpropUpdate(weights(r,c), weightGradients(r,c), prevWeightGradients(r,c), rpropWeightSteps(r,c));

    for (int i = 0; i < biasGradients.size(); ++i)
        applyRpropUpdate(biases(i), biasGradients(i), prevBiasGradients(i), rpropBiasSteps(i));

    prevWeightGradients = weightGradients.eval();
    prevBiasGradients   = biasGradients.eval();
    weightGradients.setZero();
    biasGradients.setZero();
}