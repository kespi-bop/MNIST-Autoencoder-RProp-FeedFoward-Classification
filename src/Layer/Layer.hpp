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

    size_t weightRows;
    size_t weightCols;
    size_t biasSize;

    Activation activation;

public:
    // Default constructor
    Layer() = default;

    // Constructor that allocates space for weights, biases, and assigns the activation function
    Layer(const size_t biasSize, const size_t weightRows, const size_t weightCols, Activation activationFunction);

    // Weight matrix accessor
    Eigen::MatrixXf& W() { return weight; }
    const Eigen::MatrixXf& W() const { return weight; }

    // Bias vector accessor
    Eigen::VectorXf& b() { return bias; }
    const Eigen::VectorXf& b() const { return bias; }

    // Activation function accessor
    Activation getActivation() const { return activation; }
};

#endif // LAYER_HPP