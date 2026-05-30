#pragma once

#include <string>
#include "NeuralNetwork.hpp"    // resolved via -I NeuralNetwork/ (existing build setup)

/**
 * Serializes and deserializes NeuralNetwork weights to/from binary files.
 *
 * Binary format per layer:
 *   [int32 rows][int32 cols][float32 × rows×cols]   <- weight matrix W
 *   [int32 size][float32 × size]                    <- bias vector b
 *
 * Eigen stores matrices column-major; we write/read raw data() directly,
 * so the format is endian-sensitive but perfectly portable within the same
 * machine (which is the only use-case here).
 */
namespace WeightSerializer {

    /**
     * Writes all layer weights and biases to a binary file (truncates if exists).
     * Throws std::runtime_error on any I/O failure.
     */
    void save(const NeuralNetwork& net, const std::string& path);

    /**
     * Reads weights back into a network that was constructed with the same
     * architecture.  Throws std::runtime_error if the file is missing,
     * unreadable, or does not match the expected layout.
     */
    void load(NeuralNetwork& net, const std::string& path);

    /**
     * Returns true iff the file exists AND its size is > 0.
     * Used as a pre-flight check before attempting to load weights.
     */
    bool fileExistsAndNotEmpty(const std::string& path);

} // namespace WeightSerializer
