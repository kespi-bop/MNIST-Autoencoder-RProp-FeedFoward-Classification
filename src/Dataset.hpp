#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <cstdint>
#include <fstream>

class Dataset {

private:
    std::vector<std::vector<uint8_t>> trainingImages;
    std::vector<uint8_t> trainingLabels;
    std::vector<std::vector<uint8_t>> testImages;
    std::vector<uint8_t> testLabels;

public:
    Dataset();

    const std::vector<std::vector<uint8_t>>& getTrainingImages() const;
    const std::vector<uint8_t>& getTrainingLabels() const;
    const std::vector<std::vector<uint8_t>>& getTestImages() const;
    const std::vector<uint8_t>& getTestLabels() const;

};

#endif