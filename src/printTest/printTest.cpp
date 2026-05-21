#include "printTest.hpp"
#include "Dataset.hpp"
#include <iostream>
#include <vector>
#include <cstdint>

/**
 * Renders an MNIST image to the console using ASCII characters to approximate pixel intensity.
 * It decodes the raw numeric label from the One-Hot encoding using argmax.
 * 
 * @param normalizedImg Vector of normalized float pixel values.
 * @param oneHotLabel One-Hot encoded label vector.
 * @param index Index of the image in the printing sequence.
 */
void printASCIIImage(const Eigen::VectorXf& normalizedImg, const Eigen::VectorXf& oneHotLabel, int index) {
    // Reconstruct the numeric label using argmax on the one-hot encoded vector
    Eigen::Index maxIndex;
    oneHotLabel.maxCoeff(&maxIndex);
    int label = static_cast<int>(maxIndex);

    std::cout << "========================================================\n";
    std::cout << "Image " << index + 1 << " - Associated label: " << label << "\n";
    std::cout << "One-Hot encoded label: [ " << oneHotLabel.transpose() << " ]\n";
    std::cout << "========================================================\n";
    
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            float val = normalizedImg(r * 28 + c);
            int pixel = static_cast<int>(val * 255.0f + 0.5f);
            if (pixel > 200) std::cout << "@@";
            else if (pixel > 128) std::cout << "%%";
            else if (pixel > 64) std::cout << "..";
            else std::cout << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

/**
 * Loads the MNIST dataset and prints the specified number of training and test samples
 * to the standard output. It verifies that each set was successfully loaded, ensures
 * the target print count is within bounds, and calls printASCIIImage() on each selected sample.
 * 
 * @param num_to_print The maximum number of images to render from each set.
 */
void printTest(int num_to_print){
    std::cout << "Loading dataset..." << std::endl;
    Dataset dataset;

    // --- Training Set ---
    const auto& trainingSet = dataset.getTrainingSet();

    if (trainingSet.empty()) {
        std::cerr << "Error: The training set is empty or has not been loaded correctly!" << std::endl;
        exit(1);
    }

    int real_num_to_print = std::min(num_to_print, (int)trainingSet.size());
    std::cout << "\nPrinting the first " << real_num_to_print << " images of the Training Set:\n\n";
    
    for (int i = 0; i < real_num_to_print; i++) {
        printASCIIImage(trainingSet[i].first, trainingSet[i].second, i);
    }

    // --- Test Set ---
    const auto& testSet = dataset.getTestSet();

    if (testSet.empty()) {
        std::cerr << "Error: The test set is empty or has not been loaded correctly!" << std::endl;
        exit(1);
    }

    int real_num_to_print_test = std::min(num_to_print, (int)testSet.size());
    std::cout << "\nPrinting the first " << real_num_to_print_test << " images of the Test Set:\n\n";
    
    for (int i = 0; i < real_num_to_print_test; i++) {
        printASCIIImage(testSet[i].first, testSet[i].second, i);
    }
}