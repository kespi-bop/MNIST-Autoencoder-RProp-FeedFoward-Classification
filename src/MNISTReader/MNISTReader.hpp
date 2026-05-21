#ifndef MNISTREADER_HPP
#define MNISTREADER_HPP

#include <vector>
#include <cstdint>
#include <string>

/**
 * @class MNISTReader
 * @brief Static utility class for parsing MNIST database files.
 *
 * Provides static helper methods to load, parse, and byte-swap raw binary files
 * representing MNIST digit images and their associated labels.
 */
class MNISTReader {
public:
    // Default file paths for MNIST training and test datasets.
    static constexpr const char* MNIST_TRAINING_IMAGES = "../MNIST/training set/train-images.idx3-ubyte";
    static constexpr const char* MNIST_TRAINING_LABELS = "../MNIST/training set/train-labels.idx1-ubyte";
    static constexpr const char* MNIST_TEST_IMAGES = "../MNIST/test set/t10k-images.idx3-ubyte";
    static constexpr const char* MNIST_TEST_LABELS = "../MNIST/test set/t10k-labels.idx1-ubyte";

    /**
     * @brief Reads MNIST images from a binary file.
     * @param filename Path to the raw IDX3-ubyte image file.
     * @return A 2D vector where each outer element is an image (784 bytes) and inner elements are pixels.
     * @throws std::runtime_error if file cannot be opened or if the magic number is invalid.
     */
    static std::vector<std::vector<uint8_t>> readImages(const std::string& filename);

    /**
     * @brief Reads MNIST labels from a binary file.
     * @param filename Path to the raw IDX1-ubyte label file.
     * @return A 1D vector of labels corresponding to each digit (values 0-9).
     * @throws std::runtime_error if file cannot be opened or if the magic number is invalid.
     */
    static std::vector<uint8_t> readLabels(const std::string& filename);

private:
    // Private constructor: it's a utility class, so it shouldn't be instantiated
    MNISTReader() = default;
    
    /**
     * @brief Reverses byte order (big-endian to little-endian and vice versa).
     * 
     * Since MNIST files are stored in big-endian format and x86/x64 architectures
     * are little-endian, integers read from files must be byte-swapped.
     * @param i The 32-bit integer to reverse.
     * @return The byte-swapped 32-bit integer.
     */
    static uint32_t reverseInt(uint32_t i);

    /**
     * @brief Helper function to read the standard header of an MNIST binary file.
     * 
     * Reads the magic number and the item count, and converts them to host byte order.
     * @param file Reference to an open binary ifstream.
     * @param magic_number Output reference to store the byte-swapped magic number.
     * @param count Output reference to store the byte-swapped count of items/images.
     */
    static void readHeader(std::ifstream& file, uint32_t& magic_number, uint32_t& count);
};

#endif