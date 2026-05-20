#include <iostream>
#include <vector>
#include <cstdint>
#include "Dataset.hpp"

/**
 * Renders both the raw (uint8_t) and normalized (float) versions of an MNIST image to the console
 * using ASCII characters to approximate pixel intensity.
 * 
 * It iterates through a 28x28 grid of pixels:
 * - High intensity pixels (> 200 raw, or > 200/255 normalized) are rendered as "@@"
 * - Medium intensity pixels (> 128 raw, or > 128/255 normalized) are rendered as "%%"
 * - Low intensity pixels (> 64 raw, or > 64/255 normalized) are rendered as ".."
 * - Background pixels are rendered as spaces "  "
 * 
 * @param img Vector of raw pixel values.
 * @param label The ground-truth digit label.
 * @param normalizedImg Vector of normalized float pixel values.
 * @param index Index of the image in the printing sequence.
 */
void printASCIIImage(const std::vector<uint8_t>& img, uint8_t label, const std::vector<float>& normalizedImg, int index) {
    std::cout << "========================================================\n";
    std::cout << "Image " << index + 1 << " - Associated label: " << (int)label << "\n";
    std::cout << "========================================================\n";
    
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            int pixel = img[r * 28 + c];
            if (pixel > 200) std::cout << "@@";
            else if (pixel > 128) std::cout << "%%";
            else if (pixel > 64) std::cout << "..";
            else std::cout << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "========================================================\n";
    std::cout << "Normalized image " << index + 1 << " - Associated label: " << (int)label << "\n";
    std::cout << "========================================================\n";

    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            float pixel = normalizedImg[r * 28 + c];
            if (pixel > (200.0f/255.0f)) std::cout << "@@";
            else if (pixel > (128.0f/255.0f)) std::cout << "%%";
            else if (pixel > (64.0f/255.0f)) std::cout << "..";
            else std::cout << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
}

/**
 * Loads the MNIST dataset and prints the specified number of training samples to the standard output.
 * It verifies that the dataset was successfully loaded, ensures the target print count is within
 * bounds, and calls printASCIIImage() on each selected sample.
 * 
 * @param num_to_print The maximum number of training set images to render.
 */
void printTest(int num_to_print){
    std::cout << "Loading dataset..." << std::endl;
    Dataset dataset;

    const auto& images = dataset.getTrainingImages();
    const auto& labels = dataset.getTrainingLabels();
    const auto& normalizedImages = dataset.getNormalizedTrainingImages();

    if (images.empty() || labels.empty() || normalizedImages.empty()) {
        std::cerr << "Error: The dataset is empty or has not been loaded correctly!" << std::endl;
        exit(1);
    }

    int real_num_to_print = std::min(num_to_print, (int)images.size());
    std::cout << "\nPrinting the first " << real_num_to_print << " images of the Training Set:\n\n";
    
    for (int i = 0; i < real_num_to_print; i++) {
        printASCIIImage(images[i], labels[i], normalizedImages[i], i);
    }
}