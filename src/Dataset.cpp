#include "Dataset.hpp"
#include "MNISTReader.hpp"


Dataset::Dataset() {
    trainingImages = MNISTReader::readImages(MNISTReader::MNIST_TRAINING_IMAGES);
    trainingLabels = MNISTReader::readLabels(MNISTReader::MNIST_TRAINING_LABELS);
    testImages = MNISTReader::readImages(MNISTReader::MNIST_TEST_IMAGES);
    testLabels = MNISTReader::readLabels(MNISTReader::MNIST_TEST_LABELS);
}
    
const std::vector<std::vector<uint8_t>>& Dataset::getTrainingImages() const {
    return trainingImages;
}

const std::vector<uint8_t>& Dataset::getTrainingLabels() const {
    return trainingLabels;
}

const std::vector<std::vector<uint8_t>>& Dataset::getTestImages() const {
    return testImages;
}

const std::vector<uint8_t>& Dataset::getTestLabels() const {
    return testLabels;
}