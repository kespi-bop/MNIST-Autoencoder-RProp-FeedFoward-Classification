#pragma once

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "Activation.hpp"

class Layer {
private:
    Eigen::MatrixXf weight;
    Eigen::VectorXf bias;

    Eigen::MatrixXf d_weight;       // gradient accumulator for weights
    Eigen::VectorXf d_bias;         // gradient accumulator for biases

    Eigen::MatrixXf prev_d_weight;  // previous epoch gradient (for RProp)
    Eigen::VectorXf prev_d_bias;

    Eigen::MatrixXf delta_weight;   // RProp step size per weight
    Eigen::VectorXf delta_bias;     // RProp step size per bias

    Eigen::VectorXf lastInput;
    Eigen::VectorXf a;
    Eigen::VectorXf z;

    size_t weightRows;
    size_t weightCols;
    size_t biasSize;

    Activation activation;

public:

    // Constructor that allocates space for weights, biases, and assigns the activation function
    Layer(size_t biasSize, size_t weightRows, size_t weightCols, Activation activationFunction);

    // Weight matrix accessor
    Eigen::MatrixXf& W() { return weight; }
    const Eigen::MatrixXf& W() const { return weight; }

    // Bias vector accessor
    Eigen::VectorXf& b() { return bias; }
    const Eigen::VectorXf& b() const { return bias; }

    // Weight gradient accessor
    Eigen::MatrixXf& dW() { return d_weight; }
    const Eigen::MatrixXf& dW() const { return d_weight; }

    // Bias gradient accessor
    Eigen::VectorXf& db() { return d_bias; }
    const Eigen::VectorXf& db() const { return d_bias; }

    // Pre-activation accessor (a = W*x + b)
    Eigen::VectorXf& getA() { return a; }
    const Eigen::VectorXf& getA() const { return a; }

    // Pre-activation accessor (z = W*x + b)
    Eigen::VectorXf& getZ() { return z; }
    const Eigen::VectorXf& getZ() const { return z; }

    Eigen::VectorXf& getLastInput() { return lastInput; }
    const Eigen::VectorXf& getLastInput() const { return lastInput; }

    // Activation function accessor
    Activation getActivation() const { return activation; }

    Eigen::VectorXf forward(const Eigen::VectorXf& input);

    Eigen::VectorXf backwardOutput(const Eigen::VectorXf& gradOutput);

    Eigen::VectorXf backward(Eigen::VectorXf& delta);

    void updateWeights(float learningRate);
};