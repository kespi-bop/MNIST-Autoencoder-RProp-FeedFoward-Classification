#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "Activation.hpp"

class Layer {
private:
    Eigen::MatrixXf weight;
    Eigen::VectorXf bias;

    Eigen::MatrixXf d_weight = Eigen::MatrixXf::Zero(0, 0);
    Eigen::VectorXf d_bias = Eigen::VectorXf::Zero(0);

    Eigen::VectorXf lastInput;
    Eigen::VectorXf z;
    Eigen::VectorXf a;

    size_t weightRows;
    size_t weightCols;
    size_t biasSize;

    Activation activation;

public:
    // Constructor that allocates space for weights, biases, and assigns the activation function
    Layer(const size_t biasSize, const size_t weightRows, const size_t weightCols, Activation activationFunction);

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

    // z accessor
    Eigen::VectorXf& getZ() { return z; }
    const Eigen::VectorXf& getZ() const { return z; }

    // a accessor
    Eigen::VectorXf& getA() { return a; }
    const Eigen::VectorXf& getA() const { return a; }

    Eigen::VectorXf& getLastInput() { return lastInput; }
    const Eigen::VectorXf& getLastInput() const { return lastInput; }

    // Activation function accessor
    Activation getActivation() const { return activation; }

    Eigen::VectorXf forward(const Eigen::VectorXf& input);

    Eigen::VectorXf backward(Eigen::VectorXf& delta);
};

#endif // LAYER_HPP