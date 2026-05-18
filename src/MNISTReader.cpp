#include "MNISTReader.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>

void MNISTReader::readHeader(std::ifstream& file, uint32_t& magic_number, uint32_t& count) {
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&count, sizeof(count));
    magic_number = reverseInt(magic_number);
    count = reverseInt(count);
}

std::vector<std::vector<uint8_t>> MNISTReader::readImages(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Errore: Impossibile aprire il file delle immagini " + filename);
    }

    uint32_t magic_number, number_of_images;
    readHeader(file, magic_number, number_of_images);

    if (magic_number != 2051) {
        throw std::runtime_error("Errore: Magic number non valido per le immagini in " + filename);
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

std::vector<uint8_t> MNISTReader::readLabels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Errore: Impossibile aprire il file delle label " + filename);
    }

    uint32_t magic_number, number_of_items;
    readHeader(file, magic_number, number_of_items);

    if (magic_number != 2049) {
        throw std::runtime_error("Errore: Magic number non valido per le label in " + filename);
    }

    std::vector<uint8_t> labels(number_of_items);
    file.read((char*)labels.data(), number_of_items);

    file.close();
    
    return labels;
}

uint32_t MNISTReader::reverseInt(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}
