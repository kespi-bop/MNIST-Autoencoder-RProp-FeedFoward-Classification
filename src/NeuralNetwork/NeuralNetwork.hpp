#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <Eigen/Dense>
#include <vector>
#include "Layer.hpp"

class NeuralNetwork {
protected:
    // Protected so that derived classes (Autoencoder, Classifier)
    // can directly access this vector to initialize or modify it.
    std::vector<Layer> layers;

public:
    virtual ~NeuralNetwork() = default;

    // Pure virtual methods: the core algorithm is delegated to derived classes
    virtual Eigen::VectorXf forward(const Eigen::VectorXf& input) = 0;
    virtual void backward(Eigen::VectorXf& lossGradient) = 0;

    // --- CONCRETE METHODS (inherited by all subclasses) ---

    // Returns the total number of layers in the network
    size_t getLayerCount() const { return layers.size(); }

    // Full layer vector accessor (mutable and const versions)
    std::vector<Layer>& getLayers() { return layers; }
    const std::vector<Layer>& getLayers() const { return layers; }

    // Single layer accessor by index (uses .at() for bounds checking)
    Layer& getLayer(size_t index) { return layers.at(index); }
    const Layer& getLayer(size_t index) const { return layers.at(index); }
};

#endif // NEURALNETWORK_HPP