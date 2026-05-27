#pragma once

#include <Eigen/Dense>
#include "Activation.hpp"

class Layer {
private:
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;

    Eigen::MatrixXf weightGradients;      // accumulated gradient for weights
    Eigen::VectorXf biasGradients;        // accumulated gradient for biases

    Eigen::MatrixXf prevWeightGradients;  // previous epoch gradients (RProp)
    Eigen::VectorXf prevBiasGradients;

    Eigen::MatrixXf rpropWeightSteps;     // per-weight RProp step size
    Eigen::VectorXf rpropBiasSteps;       // per-bias RProp step size

    Eigen::VectorXf cachedInput;          // saved during forward pass for backprop
    Eigen::VectorXf preActivation;        // a = W*x + b
    Eigen::VectorXf postActivation;       // z = activation(a)

    Activation activationType;

    // Shared core of backward passes: accumulates gradients and propagates upstream
    Eigen::VectorXf accumulateAndPropagate(const Eigen::VectorXf& gradient);

    // Applies RProp update rule to a single scalar parameter
    void applyRpropUpdate(float& param, float& grad, float& prevGrad, float& stepSize);

public:
    Layer(size_t outputSize, size_t weightRows, size_t inputSize, Activation activationFunction);

    Eigen::MatrixXf& W() { return weights; }
    const Eigen::MatrixXf& W() const { return weights; }

    Eigen::VectorXf& b() { return biases; }
    const Eigen::VectorXf& b() const { return biases; }

    Eigen::MatrixXf& dW() { return weightGradients; }
    const Eigen::MatrixXf& dW() const { return weightGradients; }

    Eigen::VectorXf& db() { return biasGradients; }
    const Eigen::VectorXf& db() const { return biasGradients; }

    Eigen::VectorXf& getPreActivation() { return preActivation; }
    const Eigen::VectorXf& getPreActivation() const { return preActivation; }

    Eigen::VectorXf& getPostActivation() { return postActivation; }
    const Eigen::VectorXf& getPostActivation() const { return postActivation; }

    Eigen::VectorXf& getCachedInput() { return cachedInput; }
    const Eigen::VectorXf& getCachedInput() const { return cachedInput; }

    Activation getActivationType() const { return activationType; }

    Eigen::VectorXf forward(const Eigen::VectorXf& input);

    // Backward pass for the output layer (loss gradient flows directly, no activation derivative)
    Eigen::VectorXf backwardOutputLayer(const Eigen::VectorXf& lossGradient);

    // Backward pass for hidden layers (applies activation derivative before propagating)
    Eigen::VectorXf backward(Eigen::VectorXf& incomingGradient);

    void updateWeightsRPROP(unsigned int trainingSetSize);
};