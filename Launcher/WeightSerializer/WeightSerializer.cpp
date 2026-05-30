#include "WeightSerializer.hpp"
#include <fstream>
#include <stdexcept>

namespace WeightSerializer {

// ── save ─────────────────────────────────────────────────────────────────────

void save(const NeuralNetwork& net, const std::string& path)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open())
        throw std::runtime_error("WeightSerializer::save – cannot open: " + path);

    for (const auto& layer : net.getLayers()) {
        const Eigen::MatrixXf& W = layer.W();
        const Eigen::VectorXf& b = layer.b();

        const int rows  = static_cast<int>(W.rows());
        const int cols  = static_cast<int>(W.cols());
        const int bsize = static_cast<int>(b.size());

        // Weight matrix: shape then data (column-major)
        file.write(reinterpret_cast<const char*>(&rows),  sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols),  sizeof(int));
        file.write(reinterpret_cast<const char*>(W.data()),
                   static_cast<std::streamsize>(rows * cols * sizeof(float)));

        // Bias vector: size then data
        file.write(reinterpret_cast<const char*>(&bsize), sizeof(int));
        file.write(reinterpret_cast<const char*>(b.data()),
                   static_cast<std::streamsize>(bsize * sizeof(float)));
    }

    if (!file)
        throw std::runtime_error("WeightSerializer::save – I/O error writing: " + path);
}

// ── load ─────────────────────────────────────────────────────────────────────

void load(NeuralNetwork& net, const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("WeightSerializer::load – cannot open: " + path);

    for (auto& layer : net.getLayers()) {
        int rows = 0, cols = 0, bsize = 0;

        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        if (!file)
            throw std::runtime_error("WeightSerializer::load – corrupt header in: " + path);

        layer.W().resize(rows, cols);
        file.read(reinterpret_cast<char*>(layer.W().data()),
                  static_cast<std::streamsize>(rows * cols * sizeof(float)));

        file.read(reinterpret_cast<char*>(&bsize), sizeof(int));
        if (!file)
            throw std::runtime_error("WeightSerializer::load – corrupt bias header in: " + path);

        layer.b().resize(bsize);
        file.read(reinterpret_cast<char*>(layer.b().data()),
                  static_cast<std::streamsize>(bsize * sizeof(float)));

        if (!file)
            throw std::runtime_error("WeightSerializer::load – corrupt data in: " + path);
    }
}

// ── fileExistsAndNotEmpty ─────────────────────────────────────────────────────

bool fileExistsAndNotEmpty(const std::string& path)
{
    // Open in binary mode and seek to end; if the stream is valid and the
    // position is > 0 the file exists and contains data.
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    return file.is_open() && (file.tellg() > static_cast<std::streampos>(0));
}

} // namespace WeightSerializer
