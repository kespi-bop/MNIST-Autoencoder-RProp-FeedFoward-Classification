#include "Layer.hpp"

// Constructor with pre-allocation
Layer::Layer(const size_t biasSize, const size_t weightRows, const size_t weightCols, Activation activationFunction) :
    activation(activationFunction) 
{
    bias.resize(biasSize);
    weight.resize(weightRows, weightCols);
}

