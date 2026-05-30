#pragma once

#include <string>

/**
 * Centralizes every file-name pattern used by the project.
 * All functions are inline so there is no .cpp counterpart.
 *
 * Naming convention:
 *   weights_autoencoder_h<H>.dat
 *   weights_feedforward_h<H>.dat
 *   results_test_h<H>.txt
 *
 * Using a single header avoids magic strings scattered across multiple
 * translation units and makes it trivial to change the naming policy.
 */
namespace ModelPaths {

    inline std::string autoencoderWeights(int h) {
        return "../Launcher/TrainedModels/AutoencoderWeights/weights_autoencoder_h" + std::to_string(h) + ".dat";
    }

    inline std::string feedforwardWeights(int h) {
        return "../Launcher/TrainedModels/FeedforwardWeights/weights_feedforward_h" + std::to_string(h) + ".dat";
    }

    inline std::string testResults(int h) {
        return "../Launcher/TrainedModels/TestResults/results_test_h" + std::to_string(h) + ".txt";
    }

} // namespace ModelPaths
