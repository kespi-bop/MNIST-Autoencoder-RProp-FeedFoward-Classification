#include "Dataset.hpp"
#include "MNISTReader.hpp"
#include <iostream>

Dataset::Dataset() {
    trainingImages = MNISTReader::readImages(MNISTReader::MNIST_TRAINING_IMAGES);
    trainingLabels = MNISTReader::readLabels(MNISTReader::MNIST_TRAINING_LABELS);
    testImages = MNISTReader::readImages(MNISTReader::MNIST_TEST_IMAGES);
    testLabels = MNISTReader::readLabels(MNISTReader::MNIST_TEST_LABELS);
    
    // Normalize training images to prepare them for neural network input
    normalizeImages(trainingImages);
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

const std::vector<std::vector<float>>& Dataset::getNormalizedTrainingImages() const {
    return normalizedTrainingImages;
}

/**
 * Normalizes raw pixel values from their original [0, 255] integer range to [0.0, 1.0] float range.
 * This preprocessing step is vital for neural network stability and efficient gradient descent.
 * The method resizes the output normalized image matrix to match the input dimensions and converts
 * each byte value to a float divided by 255.0.
 */
void Dataset::normalizeImages(const std::vector<std::vector<uint8_t>>& imagesToNormalize) {

    normalizedTrainingImages.resize(imagesToNormalize.size());

    for (int i = 0; i < imagesToNormalize.size(); i++) {
        normalizedTrainingImages[i].resize(imagesToNormalize[i].size());
        for (int j = 0; j < imagesToNormalize[i].size(); j++) {
            normalizedTrainingImages[i][j] = imagesToNormalize[i][j] / 255.0f;
        }
    }

    std::cout << "Preprocessing completato: immagini normalizzate!" << std::endl;
    
}