#include "MNISTReader.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>

/**
 * Reads a standard 8-byte header from the MNIST binary file stream,
 * consisting of a 4-byte magic number and a 4-byte item/image count.
 * Because MNIST files are stored in Big-Endian format, both values are
 * byte-swapped to match the host platform's Endianness (usually Little-Endian on x86/x64).
 */
void MNISTReader::readHeader(std::ifstream& file, uint32_t& magic_number, uint32_t& count) {
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&count, sizeof(count));
    magic_number = reverseInt(magic_number);
    count = reverseInt(count);
}

/**
 * Opens the specified IDX3-ubyte file and parses the entire set of MNIST images.
 * 
 * The function performs the following steps:
 * 1. Opens the file in binary mode and verifies it is successfully opened.
 * 2. Reads and validates the header using readHeader(). IDX3 files must start with magic number 2051.
 * 3. Reads the 2D dimensions of the images (rows and columns) and converts them to host byte order.
 * 4. Computes the raw size of a single image (rows * columns).
 * 5. Sequentially reads the raw pixel data for each image into a dynamically allocated vector of bytes
 *    and appends them to the result vector.
 * 
 * @param filename File path to the IDX3-ubyte images dataset.
 * @return A 2D vector containing the deserialized images.
 */
std::vector<std::vector<uint8_t>> MNISTReader::readImages(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open image file " + filename);
    }

    uint32_t magic_number, number_of_images;
    readHeader(file, magic_number, number_of_images);

    if (magic_number != 2051) {
        throw std::runtime_error("Error: Invalid magic number for images in " + filename);
    }

    uint32_t n_rows = 0;
    uint32_t n_cols = 0;
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));

    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);

    size_t image_size = n_rows * n_cols;
    std::vector<std::vector<uint8_t>> images;
    images.reserve(number_of_images);

    for (size_t i = 0; i < number_of_images; i++) {
        std::vector<uint8_t> image(image_size);
        file.read((char*)image.data(), image_size);     
        images.push_back(std::move(image));
    }
    
    file.close();
    
    return images;
}

/**
 * Opens the specified IDX1-ubyte file and parses the entire set of MNIST labels.
 * 
 * The function performs the following steps:
 * 1. Opens the file in binary mode and verifies it is successfully opened.
 * 2. Reads and validates the header using readHeader(). IDX1 files must start with magic number 2049.
 * 3. Dynamically allocates a vector sized to hold all labels, and reads the raw byte array directly from the file.
 * 
 * @param filename File path to the IDX1-ubyte labels dataset.
 * @return A 1D vector containing the deserialized labels.
 */
std::vector<uint8_t> MNISTReader::readLabels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open label file " + filename);
    }

    uint32_t magic_number, number_of_items;
    readHeader(file, magic_number, number_of_items);

    if (magic_number != 2049) {
        throw std::runtime_error("Error: Invalid magic number for labels in " + filename);
    }

    std::vector<uint8_t> labels(number_of_items);
    file.read((char*)labels.data(), number_of_items);

    file.close();
    
    return labels;
}

/**
 * Reverses the byte order of a 32-bit unsigned integer.
 * This is used to convert 32-bit integers from Big-Endian (file format) to Little-Endian (host platform).
 * It extracts each of the 4 bytes using bit-shifts and masks, and reassembles them in reverse order.
 */
uint32_t MNISTReader::reverseInt(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}
