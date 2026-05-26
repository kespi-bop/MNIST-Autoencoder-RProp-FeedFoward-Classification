#include "Dataset.hpp"
#include "MNISTReader.hpp"
#include <iostream>
#include <algorithm>
#include <random>

Dataset::Dataset() {
    // Normalize images to prepare them for neural network input
    fullTrainingSetImages = normalizeImages(MNISTReader::readImages(MNISTReader::MNIST_TRAINING_IMAGES));
    testSetImages = normalizeImages(MNISTReader::readImages(MNISTReader::MNIST_TEST_IMAGES));

    // Convert labels to One-Hot encoding representation
    fullTrainingSetLabels = oneHotEncodeLabels(MNISTReader::readLabels(MNISTReader::MNIST_TRAINING_LABELS));
    testSetLabels = oneHotEncodeLabels(MNISTReader::readLabels(MNISTReader::MNIST_TEST_LABELS));

    verifyDimensions();
    splitDataset();

    // Debug: stampa statistiche sui set
    std::cout << "Training set: " << trainingSet.size() << " examples\n";
    std::cout << "Validation set: " << validationSet.size() << " examples\n";
    std::cout << "Test set: " << testSet.size() << " examples\n";
}
    
const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& Dataset::getTrainingSet() const {
    return trainingSet;
}

const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& Dataset::getValidationSet() const {
    return validationSet;
}

const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& Dataset::getTestSet() const {
    return testSet;
}

/**
 * Normalizes raw pixel values from their original [0, 255] integer range to [0.0, 1.0] float range.
 * This preprocessing step is vital for neural network stability and efficient gradient descent.
 * The method resizes the output normalized image matrix to match the input dimensions and converts
 * each byte value to a float divided by 255.0.
 */
std::vector<Eigen::VectorXf> Dataset::normalizeImages(const std::vector<std::vector<uint8_t>>& imagesToNormalize) {

    std::vector<Eigen::VectorXf> normalized;
    normalized.resize(imagesToNormalize.size());

    for (size_t i = 0; i < imagesToNormalize.size(); i++) {
        normalized[i].resize(imagesToNormalize[i].size());
        for (size_t j = 0; j < imagesToNormalize[i].size(); j++) {
            normalized[i](j) = imagesToNormalize[i][j] / 255.0f;
        }
    }
    std::cout << "Completed preprocessing, the dataset is now normalized!" << std::endl;
    return normalized;
    
}

/**
 * Converts raw numeric labels (0-9) to a One-Hot encoding representation as Eigen vectors.
 * This is essential for training neural networks for classification.
 */
std::vector<Eigen::VectorXf> Dataset::oneHotEncodeLabels(const std::vector<uint8_t>& labelsToEncode) {

    std::vector<Eigen::VectorXf> oneHot;
    oneHot.resize(labelsToEncode.size());

    for (size_t i = 0; i < labelsToEncode.size(); i++) {
        // MNIST has 10 classes (digits 0-9)
        oneHot[i] = Eigen::VectorXf::Zero(10);
        uint8_t label = labelsToEncode[i];
        if (label < 10) {
            oneHot[i](label) = 1.0f;
        } else {
            std::cerr << "Error: Label out of bounds: " << (int)label << std::endl;
        }
    }

    std::cout << "Completed preprocessing, the dataset is now one-hot encoded!" << std::endl;
    return oneHot;
    
}

void Dataset::splitDataset() {
    // Shuffle the dataset
    std::vector<size_t> indices(fullTrainingSetImages.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

    std::vector<size_t> testIndices(testSetImages.size());
    std::iota(testIndices.begin(), testIndices.end(), 0);
    std::shuffle(testIndices.begin(), testIndices.end(), std::mt19937{std::random_device{}()});

    for (size_t i = 0; i < trainSetSize; i++)
        trainingSet.emplace_back(fullTrainingSetImages[indices[i]], fullTrainingSetLabels[indices[i]]);

    for (size_t i = trainSetSize; i < trainSetSize + validSetSize; i++)
        validationSet.emplace_back(fullTrainingSetImages[indices[i]], fullTrainingSetLabels[indices[i]]);

    for (size_t i = 0; i < testSetSize; i++)
        testSet.emplace_back(testSetImages[testIndices[i]], testSetLabels[testIndices[i]]);
}


void Dataset::verifyDimensions() {
    if (fullTrainingSetImages.size() < fullTrainSetSize)
        throw std::runtime_error("Dataset error: not enough training images (expected " 
            + std::to_string(fullTrainSetSize) + ", got " + std::to_string(fullTrainingSetImages.size()) + ")");

    if (testSetImages.size() < testSetSize)
        throw std::runtime_error("Dataset error: not enough test images (expected " 
            + std::to_string(testSetSize) + ", got " + std::to_string(testSetImages.size()) + ")");

    if (fullTrainingSetLabels.size() < fullTrainSetSize)
        throw std::runtime_error("Dataset error: not enough training labels (expected " 
            + std::to_string(fullTrainSetSize) + ", got " + std::to_string(fullTrainingSetLabels.size()) + ")");

    if (testSetLabels.size() < testSetSize)
        throw std::runtime_error("Dataset error: not enough test labels (expected " 
            + std::to_string(testSetSize) + ", got " + std::to_string(testSetLabels.size()) + ")");
}