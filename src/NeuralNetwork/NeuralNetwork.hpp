#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"

class NeuralNetwork {
protected:
    std::vector<Layer> layers;

public:
    virtual ~NeuralNetwork() = default;

    virtual Eigen::VectorXf forward(const Eigen::VectorXf& input) = 0;
    virtual void backward(Eigen::VectorXf& lossGradient) = 0;
    virtual void updateWeightsRPROP(unsigned int trainingSetSize) = 0;

    std::vector<Layer>& getLayers() { return layers; }
    const std::vector<Layer>& getLayers() const { return layers; }
};

#endif // NEURALNETWORK_HPP