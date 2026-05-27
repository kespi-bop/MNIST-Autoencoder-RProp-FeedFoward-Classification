#include "Dataset.hpp"
#include "NeuralNetwork/Autoencoder/Autoencoder.hpp"
#include "Trainer/Trainer.hpp"
#include "Trainer/LossFunction/LossFunction.hpp"

int main() {
    Dataset dataset;
    const auto& trainingSet   = dataset.getTrainingSet();
    const auto& validationSet = dataset.getValidationSet();

    Autoencoder autoencoder(trainingSet[0].first.size(), 25);
    Trainer trainer(100, 10, autoencoder, LossFunction::BinaryCrossEntropy, trainingSet, validationSet);
    trainer.train();

    return 0;
}