#include "TrainingPipeline.hpp"

#include "Autoencoder.hpp"
#include "FeedForward.hpp"
#include "Trainer.hpp"
#include "LossFunction.hpp"
#include "ModelPaths.hpp"
#include "WeightSerializer.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>    // std::max_element
#include <stdexcept>

// ── Types ─────────────────────────────────────────────────────────────────────

namespace {

using DataPairs = std::vector<std::pair<Eigen::VectorXf, Eigen::VectorXf>>;

// ── Private helpers (translation-unit scope) ──────────────────────────────────

/**
 * Builds an autoencoder-style dataset where the target is identical to the
 * input (reconstruction task).  Returns a new vector; source is not modified.
 */
DataPairs makeReconstructionPairs(const DataPairs& source)
{
    DataPairs result;
    result.reserve(source.size());
    for (const auto& [image, _] : source)
        result.push_back({image, image});
    return result;
}

/**
 * Encodes every sample in source through the encoder layer of ae, preserving
 * the original label.  Returns a new vector; source is not modified.
 */
DataPairs buildEncodedPairs(Autoencoder& ae, const DataPairs& source)
{
    DataPairs result;
    result.reserve(source.size());
    for (const auto& [input, label] : source)
        result.push_back({ae.getEncoding(input), label});
    return result;
}

/**
 * Calculate the reconstruction error of the autoencoder on the test set.
 */
float computeReconstructionMSE(Autoencoder& ae, const DataPairs& testSet)
{
    float totalError = 0.0f;
    for (const auto& [input, _] : testSet) {
        // Supponendo che ae abbia un metodo forward o un modo per ottenere la ricostruzione completa
        // Se non hai un metodo .forward() globale sull'Autoencoder, usa il meccanismo previsto dalla tua classe
        Eigen::VectorXf reconstructed = ae.forward(input); 
        totalError += (input - reconstructed).squaredNorm() / static_cast<float>(input.size());
    }
    return totalError / static_cast<float>(testSet.size());
}

/**
 * Computes classification accuracy (%) over a set of (encoded_input, one-hot_label)
 * pairs by argmax comparison.
 */
float computeAccuracy(FeedForward& classifier, const DataPairs& testSet)
{
    int correct = 0;
    for (const auto& [input, target] : testSet) {
        const Eigen::VectorXf output = classifier.forward(input);

        const int predicted = static_cast<int>(
            std::distance(output.data(),
                          std::max_element(output.data(), output.data() + output.size())));
        const int actual = static_cast<int>(
            std::distance(target.data(),
                          std::max_element(target.data(), target.data() + target.size())));

        if (predicted == actual) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(testSet.size()) * 100.0f;
}

/**
 * Writes accuracy to the result file and also prints it.
 * Throws std::runtime_error if the file cannot be opened.
 */
void saveTestResults(int h, float reconstructionError, float accuracy)
{
    const std::string path = ModelPaths::testResults(h);
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open())
        throw std::runtime_error("TrainingPipeline::saveTestResults – cannot write: " + path);

    out << "h = " << h << "\n"
        << "Reconstruction Error (Test): " << reconstructionError << "\n"
        << "Classification Accuracy: " << accuracy << "%\n";

    std::cout << "[h=" << h << "] Reconstruction Error: " << reconstructionError << "\n"
              << "[h=" << h << "] Test Accuracy: " << accuracy << "%\n"
              << "→ saved to: " << path << "\n";
}

/**
 * Validates that the weight file for a given tag exists and is non-empty.
 * Throws std::runtime_error with a descriptive message on failure.
 */
void requireWeightFile(const std::string& path, const std::string& modelTag)
{
    if (!WeightSerializer::fileExistsAndNotEmpty(path))
        throw std::runtime_error(
            "Prerequisito mancante: il file " + modelTag + " è assente o vuoto.\n"
            "  Percorso atteso: " + path + "\n"
            "  Esegui prima l'addestramento corrispondente.");
}

} // anonymous namespace

// ── Public pipeline implementations ──────────────────────────────────────────

namespace TrainingPipeline {

void trainAutoencoder(int h, const Dataset& dataset)
{
    std::cout << "\n[h=" << h << "] ════ Avvio addestramento Autoencoder ════\n";

    const DataPairs& trainSet = dataset.getTrainingSet();
    const DataPairs& validSet = dataset.getValidationSet();

    if (trainSet.empty())
        throw std::runtime_error("trainAutoencoder: il training set è vuoto.");

    const DataPairs aeTrainSet = makeReconstructionPairs(trainSet);
    const DataPairs aeValidSet = makeReconstructionPairs(validSet);

    const size_t inputSize = static_cast<size_t>(trainSet[0].first.size());
    Autoencoder autoencoder(inputSize, static_cast<size_t>(h));

    Trainer trainer(
        /*maxEpochs=*/  100,
        /*patience=*/   10,
        autoencoder,
        LossFunction::BinaryCrossEntropy,
        aeTrainSet,
        aeValidSet
    );
    trainer.train();

    const std::string weightsPath = ModelPaths::autoencoderWeights(h);
    WeightSerializer::save(autoencoder, weightsPath);
    std::cout << "[h=" << h << "] Pesi Autoencoder salvati in: " << weightsPath << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────

void trainFeedforward(int h, const Dataset& dataset)
{
    std::cout << "\n[h=" << h << "] ════ Avvio addestramento FeedForward ════\n";

    const std::string aeWeightsPath = ModelPaths::autoencoderWeights(h);
    requireWeightFile(aeWeightsPath, "Autoencoder (h=" + std::to_string(h) + ")");

    const DataPairs& trainSet = dataset.getTrainingSet();
    const DataPairs& validSet = dataset.getValidationSet();

    if (trainSet.empty())
        throw std::runtime_error("trainFeedforward: il training set è vuoto.");

    // Load the already-trained encoder
    const size_t inputSize = static_cast<size_t>(trainSet[0].first.size());
    Autoencoder autoencoder(inputSize, static_cast<size_t>(h));
    WeightSerializer::load(autoencoder, aeWeightsPath);
    std::cout << "[h=" << h << "] Pesi Autoencoder caricati da: " << aeWeightsPath << "\n";

    // Encode the dataset once; reuse for all training epochs
    const DataPairs encodedTrain = buildEncodedPairs(autoencoder, trainSet);
    const DataPairs encodedValid = buildEncodedPairs(autoencoder, validSet);

    FeedForward classifier(static_cast<size_t>(h), 256, 10);

    Trainer trainer(
        /*maxEpochs=*/  200,
        /*patience=*/   20,
        classifier,
        LossFunction::CategoricalCrossEntropy,
        encodedTrain,
        encodedValid
    );
    trainer.train();

    const std::string ffWeightsPath = ModelPaths::feedforwardWeights(h);
    WeightSerializer::save(classifier, ffWeightsPath);
    std::cout << "[h=" << h << "] Pesi FeedForward salvati in: " << ffWeightsPath << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────

void trainFull(int h, const Dataset& dataset)
{
    trainAutoencoder(h, dataset);
    trainFeedforward(h, dataset);
}

// ─────────────────────────────────────────────────────────────────────────────

void runTest(int h, const Dataset& dataset)
{
    std::cout << "\n[h=" << h << "] ════ Avvio Test Set ════\n";

    const std::string aeWeightsPath = ModelPaths::autoencoderWeights(h);
    const std::string ffWeightsPath = ModelPaths::feedforwardWeights(h);

    requireWeightFile(aeWeightsPath, "Autoencoder (h=" + std::to_string(h) + ")");
    requireWeightFile(ffWeightsPath, "FeedForward (h=" + std::to_string(h) + ")");

    const DataPairs& trainSet = dataset.getTrainingSet();  // used for inputSize
    const DataPairs& testSet  = dataset.getTestSet();

    if (trainSet.empty() || testSet.empty())
        throw std::runtime_error("runTest: training set o test set vuoti.");

    // Reconstruct and load both models
    const size_t inputSize = static_cast<size_t>(trainSet[0].first.size());

    Autoencoder autoencoder(inputSize, static_cast<size_t>(h));
    WeightSerializer::load(autoencoder, aeWeightsPath);
    std::cout << "[h=" << h << "] Pesi Autoencoder caricati da: " << aeWeightsPath << "\n";

    FeedForward classifier(static_cast<size_t>(h), 256, 10);
    WeightSerializer::load(classifier, ffWeightsPath);
    std::cout << "[h=" << h << "] Pesi FeedForward caricati da: " << ffWeightsPath << "\n";

    const float reconErr = computeReconstructionMSE(autoencoder, testSet);

    // Encode test set through the loaded autoencoder
    const DataPairs encodedTest = buildEncodedPairs(autoencoder, testSet);

    const float accuracy = computeAccuracy(classifier, encodedTest);
    saveTestResults(h, reconErr, accuracy);
}

} // namespace TrainingPipeline
