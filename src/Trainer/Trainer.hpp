#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "NeuralNetwork.hpp"
#include "LossFunctions.hpp"

class Trainer {
public:
    Trainer(int maxEpochs, int patience, NeuralNetwork& model,
            LossFunction lossFunction,
            const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& trainingSet,
            const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& validationSet)
        : maxEpochs(maxEpochs), patience(patience), model(model),
          lossFunction(lossFunction), trainingSet(trainingSet), validationSet(validationSet) {}

    void train();

private:
    int          maxEpochs;
    int          patience;
    NeuralNetwork& model;
    LossFunction   lossFunction;
    const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> trainingSet;
    const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> validationSet;

    float computeValidationLoss();
    float runTrainingEpoch();
    float computeReconstructionError();

    static void appendLayerSnapshot(
        std::vector<Eigen::MatrixXf>& weightHistory,
        std::vector<Eigen::VectorXf>& biasHistory,
        const std::vector<Layer>& layers);
};

#endif // TRAINER_HPP