#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

// Function to convert from Big-Endian (MNIST) to Little-Endian (CPU x86/ARM)
uint32_t reverseInt(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

int main() {
    // Extracted MNIST filename
    std::string filename = "../MNIST/training set/train-images.idx3-ubyte";
    std::string labels_filename = "../MNIST/training set/train-labels.idx1-ubyte";
    
    // Open file in binary mode
    std::ifstream file(filename, std::ios::binary);
    std::ifstream labels_file(labels_filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file '" << filename << "'." << std::endl;
        std::cerr << "Make sure it is extracted and in the same folder as the executable." << std::endl;
        return 1;
    }
    if (!labels_file.is_open()) {
        std::cerr << "Error: Unable to open file '" << labels_filename << "'." << std::endl;
        return 1;
    }

    // Variables for the file header
    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t n_rows = 0;
    uint32_t n_cols = 0;

    // Read the first 4 fields of the header (4 bytes each)
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));

    uint32_t labels_magic_number = 0;
    uint32_t number_of_labels = 0;
    labels_file.read((char*)&labels_magic_number, sizeof(labels_magic_number));
    labels_file.read((char*)&number_of_labels, sizeof(number_of_labels));

    // Convert values from MNIST endianness to CPU endianness
    magic_number = reverseInt(magic_number);
    number_of_images = reverseInt(number_of_images);
    n_rows = reverseInt(n_rows);
    n_cols = reverseInt(n_cols);

    labels_magic_number = reverseInt(labels_magic_number);
    number_of_labels = reverseInt(number_of_labels);

    // Security check on Magic Number (for MNIST images it must be 2051)
    if (magic_number != 2051) {
        std::cerr << "Error: Invalid magic number. The file might be corrupted or is not the image file." << std::endl;
        return 1;
    }

    std::cout << "--- DATASET INFO ---" << std::endl;
    std::cout << "Magic Number: " << magic_number << std::endl;
    std::cout << "Total Images: " << number_of_images << std::endl;
    std::cout << "Resolution: " << n_rows << "x" << n_cols << std::endl;
    std::cout << "--------------------" << std::endl << std::endl;

    // Size of a single image in bytes (28 * 28 = 784)
    size_t image_size = n_rows * n_cols;
    std::vector<unsigned char> image(image_size);


    for(int i=0; i < 10; i++){
        // Read the bytes of the image
        file.read((char*)image.data(), image_size);
        
        // Read the label
        unsigned char label;
        labels_file.read((char*)&label, 1);
        
        std::cout << "Displaying image number: " << i << " - Label: " << (int)label << " (ASCII Art):" << std::endl;
        
        // Loop to print the 28x28 matrix on the terminal
        for (size_t r = 0; r < n_rows; ++r) {
            for (size_t c = 0; c < n_cols; ++c) {
                // Linear index in the dynamic vector
                unsigned char pixel = image[r * n_cols + c];
                // Choose an ASCII character based on pixel intensity (0-255)
                if (pixel == 0) {
                    std::cout << "  ";  // Black background (empty)
                } else if (pixel < 128) {
                    std::cout << "..";  // Dark gray
                } else {
                    std::cout << "##";  // White/Light gray
                }
            }
            std::cout << std::endl; // New row of pixels
        }
    }
    // Close the files
    file.close();
    labels_file.close();

    return 0;
}