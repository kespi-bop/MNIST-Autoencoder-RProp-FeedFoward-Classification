#include "printTest.hpp"
#include "Dataset.hpp"
#include "NeuralNetwork/Autoencoder/Autoencoder.hpp"
#include "Trainer/Trainer.hpp"
#include "Trainer/LossFunction/LossFunction.hpp"


int main() {
    
    // printTest(5);

    Dataset dataset;
    const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& trainingSet = dataset.getTrainingSet();

    Autoencoder autoencoder(trainingSet[0].first.size(),25);
    Trainer trainer(100, 0.5f, autoencoder, LossFunction::BinaryCrossEntropy, dataset.getTrainingSet());
    trainer.train();

    return 0;
}