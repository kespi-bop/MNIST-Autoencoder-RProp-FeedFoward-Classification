#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <cstdint>
#include <Eigen/Dense>

class Dataset {

public:
    Dataset();

    const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& getTrainingSet() const;
    const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& getValidationSet() const;
    const std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>& getTestSet() const;

private:
    // Data set sizes 
    static constexpr const size_t fullTrainSetSize = 60000;
    static constexpr const size_t trainSetSize = 50000;
    static constexpr const size_t validSetSize = 10000;
    static constexpr const size_t testSetSize = 10000;

    std::vector<Eigen::VectorXf> fullTrainingSetImages;
    std::vector<Eigen::VectorXf> fullTrainingSetLabels;
    std::vector<Eigen::VectorXf> testSetImages;
    std::vector<Eigen::VectorXf> testSetLabels;

    std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> trainingSet;
    std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> validationSet;
    std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>> testSet;


    std::vector<Eigen::VectorXf> normalizeImages(const std::vector<std::vector<uint8_t>>& imagesToNormalize);
    std::vector<Eigen::VectorXf> oneHotEncodeLabels(const std::vector<uint8_t>& labelsToEncode);

    void verifyDimensions();
    void splitDataset();

};

#endif