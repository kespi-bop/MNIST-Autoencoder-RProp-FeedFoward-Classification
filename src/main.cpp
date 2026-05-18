#include <iostream>
#include <vector>
#include <cstdint>
#include "Dataset.hpp"

// Funzione di utilità per stampare un'immagine in ASCII art
void printASCIIImage(const std::vector<uint8_t>& img, uint8_t label, int index) {
    std::cout << "========================================================\n";
    std::cout << "Immagine " << index + 1 << " - Label associata: " << (int)label << "\n";
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
}

int main() {
    std::cout << "Caricamento dataset in corso..." << std::endl;
    Dataset dataset;

    const auto& images = dataset.getTrainingImages();
    const auto& labels = dataset.getTrainingLabels();

    if (images.empty() || labels.empty()) {
        std::cerr << "Errore: Il dataset è vuoto o non è stato caricato correttamente!" << std::endl;
        return 1;
    }

    int num_to_print = std::min(10, (int)images.size());
    std::cout << "\nStampo le prime " << num_to_print << " immagini del Training Set:\n\n";
    
    for (int i = 0; i < num_to_print; i++) {
        printASCIIImage(images[i], labels[i], i);
    }

    return 0;
}