#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "NeuralNetwork.hpp"
#include "LossFunctions.hpp"

class Trainer{
    private:
        int maxEpochs;
        float learningRate;
        NeuralNetwork& model;
        LossFunction lossFunction;
        const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> trainingSet;
        
        void trainEpoch(int epoch);

    public:
        Trainer(int maxEpochs, float learningRate, NeuralNetwork& model, LossFunction lossFunction, const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& trainingSet) : 
            maxEpochs(maxEpochs), learningRate(learningRate), model(model), lossFunction(lossFunction), trainingSet(trainingSet) {};
        void train();
};

#endif // TRAINER_HPP