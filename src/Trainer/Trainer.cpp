#include "Trainer.hpp"
#include <iostream>

void Trainer::trainEpoch(int epoch) {
    float loss = 0.0f;
    for (size_t i = 0; i < trainingSet.size(); i++) {
        const Eigen::VectorXf& target = trainingSet[i].first;
        Eigen::VectorXf output = model.forward(target);

        Eigen::VectorXf lossGradient = LossFunctions::derivative(lossFunction, output, target);
        Eigen::VectorXf lossVec = LossFunctions::apply(lossFunction, output, target);

        loss -= lossVec.sum() / static_cast<float>(target.size());
        model.backward(lossGradient);
    }
    loss /= static_cast<float>(trainingSet.size());
    model.updateWeights(learningRate);
    std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
}

void Trainer::train() {
    for (int epoch = 0; epoch < maxEpochs; epoch++) {
        trainEpoch(epoch);
    }
}