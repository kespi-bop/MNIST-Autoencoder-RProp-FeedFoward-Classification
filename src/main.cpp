#include "Dataset.hpp"
#include "NeuralNetwork/Autoencoder/Autoencoder.hpp"
#include "NeuralNetwork/FeedForward/FeedForward.hpp"
#include "Trainer/Trainer.hpp"
#include "Trainer/LossFunction/LossFunction.hpp"
#include <iostream>

int main() {
    Dataset dataset;
    const auto& trainingSet   = dataset.getTrainingSet();
    const auto& validationSet = dataset.getValidationSet();
    const auto& testSet       = dataset.getTestSet();
    std::vector<int> h = {25,50,75,100,125};

    //INIZIO PARALLELISMO OMP per gli h = {25,50,75,100,125}

    // Per l'autoencoder il target è l'input stesso (ricostruzione)
    std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> autoencoderTrainSet;
    std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> autoencoderValidSet;
    autoencoderTrainSet.reserve(trainingSet.size());
    autoencoderValidSet.reserve(validationSet.size());
    for (const auto& [image, _] : trainingSet)
        autoencoderTrainSet.push_back({image, image});
    for (const auto& [image, _] : validationSet)
        autoencoderValidSet.push_back({image, image});

    Autoencoder autoencoder(trainingSet[0].first.size(), h[4]);
    Trainer autoencoderTrainer(100, 10, autoencoder, LossFunction::BinaryCrossEntropy, autoencoderTrainSet, autoencoderValidSet);
    autoencoderTrainer.train();

    std::cout << "Reconstruction Error: " << autoencoderTrainer.computeReconstructionError() << "\n";
    std::cout << "FINE ADDESTRAMENTO AUTOENCODER con h=" << h[4] << ", \nINIZIO ADDESTRAMENTO CLASSIFICATION" << std::endl;

    std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> encodedTrainSet;
    std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> encodedValidationSet;
    std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> encodedTestSet;

    encodedTrainSet.reserve(trainingSet.size());
    encodedValidationSet.reserve(validationSet.size());
    encodedTestSet.reserve(testSet.size());

    for (const auto& [input, label] : trainingSet) {
        encodedTrainSet.push_back({autoencoder.getEncoding(input), label});
    }

    for (const auto& [input, label] : validationSet) {
        encodedValidationSet.push_back({autoencoder.getEncoding(input), label});
    }

    for (const auto& [input, label] : testSet) {
        encodedTestSet.push_back({autoencoder.getEncoding(input), label});
    }

    FeedForward classifier(h[4], 256, 10);
    Trainer classifierTrainer(200, 20, classifier, LossFunction::CategoricalCrossEntropy, encodedTrainSet, encodedValidationSet);
    classifierTrainer.train();

    std::cout << "Accuracy del classificatore: " << classifierTrainer.computeAccuracy(encodedTestSet) << "%\n";

    //FINE PARALLELISMO OMP

    return 0;
}