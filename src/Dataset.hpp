#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <cstdint>
#include <fstream>

class Dataset {

public:
    Dataset();

    const std::vector<std::vector<uint8_t>>& getTrainingImages() const;
    const std::vector<uint8_t>& getTrainingLabels() const;
    const std::vector<std::vector<uint8_t>>& getTestImages() const;
    const std::vector<uint8_t>& getTestLabels() const;
    const std::vector<std::vector<float>>& getNormalizedTrainingImages() const;

private:
    std::vector<std::vector<uint8_t>> trainingImages;
    std::vector<uint8_t> trainingLabels;
    std::vector<std::vector<uint8_t>> testImages;
    std::vector<uint8_t> testLabels;
    std::vector<std::vector<float>> normalizedTrainingImages;

    void normalizeImages(const std::vector<std::vector<uint8_t>>& imagesToNormalize);

};

#endif