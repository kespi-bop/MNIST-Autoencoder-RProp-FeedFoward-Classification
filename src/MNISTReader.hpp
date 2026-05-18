#ifndef MNISTREADER_HPP
#define MNISTREADER_HPP

#include <vector>
#include <cstdint>
#include <string>

class MNISTReader {
public:
    static constexpr const char* MNIST_TRAINING_IMAGES = "../MNIST/training set/train-images.idx3-ubyte";
    static constexpr const char* MNIST_TRAINING_LABELS = "../MNIST/training set/train-labels.idx1-ubyte";
    static constexpr const char* MNIST_TEST_IMAGES = "../MNIST/test set/t10k-images.idx3-ubyte";
    static constexpr const char* MNIST_TEST_LABELS = "../MNIST/test set/t10k-labels.idx1-ubyte";

    static std::vector<std::vector<uint8_t>> readImages(const std::string& filename);
    static std::vector<uint8_t> readLabels(const std::string& filename);

private:
    // Costruttore privato: è una classe di sola utilità, non va istanziata
    MNISTReader() = default;
    
    static uint32_t reverseInt(uint32_t i);
    static void readHeader(std::ifstream& file, uint32_t& magic_number, uint32_t& count);
};

#endif