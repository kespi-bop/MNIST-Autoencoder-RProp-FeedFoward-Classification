#include "Trainer.hpp"
#include <iostream>

void Trainer::trainEpoch(int epoch) {
    for(int i = 0; i < trainingSet.size(); i++) {
        if(i % 1000 == 0) {
            std::cout << "Epoch " << epoch << " Iteration " << i << std::endl;
        } 
        if (i == 5000) {
            exit(0);
        }
        Eigen::VectorXf output = model.forward(trainingSet[i].first);
        Eigen::VectorXf lossGradient = LossFunctions::derivative(lossFunction, trainingSet[i].first, output, trainingSet[i].second);
        model.backward(lossGradient);
    }
}

void Trainer::train() {
    for(int epoch = 0; epoch < maxEpochs; epoch++) {
        trainEpoch(epoch);
        // float validationLoss = 0;
        // for(int i = 0; i < validationSet.size(); i++) {
        //     Eigen::VectorXf output = model.forward(validationSet[i].first);
        //     validationLoss += LossFunction::calculate(lossFunction, output, validationSet[i].second);
        // }
        // validationLoss /= validationSet.size();
        // if(validationLoss < bestValidationLoss) {
        //     bestValidationLoss = validationLoss;
        //     model.save(modelPath);
        // }
    }
}