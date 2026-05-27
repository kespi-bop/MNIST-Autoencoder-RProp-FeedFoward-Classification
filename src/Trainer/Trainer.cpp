#include "Trainer.hpp"
#include <iostream>

void Trainer::appendLayerSnapshot(
    std::vector<Eigen::MatrixXf>& weightHistory,
    std::vector<Eigen::VectorXf>& biasHistory,
    const std::vector<Layer>& layers)
{
    for (const auto& layer : layers) {
        weightHistory.push_back(layer.W());
        biasHistory.push_back(layer.b());
    }
}

float Trainer::computeReconstructionError() {
    float totalError = 0.0f;

    for (const auto& [input, _] : validationSet) {
        const Eigen::VectorXf output = model.forward(input);
        totalError += (output - input).squaredNorm() / input.size();
    }

    return totalError / validationSet.size();
}

float Trainer::computeValidationLoss() {
    float totalLoss = 0.0f;
    for (const auto& [target, _] : validationSet) {
        const Eigen::VectorXf output  = model.forward(target);
        const Eigen::VectorXf lossVec = LossFunctions::apply(lossFunction, output, target);
        totalLoss -= lossVec.sum() / static_cast<float>(target.size());
    }
    return totalLoss / static_cast<float>(validationSet.size());
}

float Trainer::runTrainingEpoch() {
    float totalLoss = 0.0f;
    for (const auto& [target, _] : trainingSet) {
        const Eigen::VectorXf output       = model.forward(target);
        Eigen::VectorXf       lossGradient = LossFunctions::derivative(lossFunction, output, target);
        const Eigen::VectorXf lossVec      = LossFunctions::apply(lossFunction, output, target);

        totalLoss -= lossVec.sum() / static_cast<float>(target.size());
        model.backward(lossGradient);
    }
    return totalLoss / static_cast<float>(trainingSet.size());
}

void Trainer::train() {
    float bestValidationLoss      = std::numeric_limits<float>::max();
    int   bestEpoch               = 0;
    int   epochsWithoutImprovement = 0;

    std::vector<Layer>           layers = model.getLayers();
    std::vector<Eigen::MatrixXf> bestWeights;
    std::vector<Eigen::VectorXf> bestBiases;
    appendLayerSnapshot(bestWeights, bestBiases, layers);

    for (int epoch = 0; epoch < maxEpochs; ++epoch) {
        const float trainingLoss   = runTrainingEpoch();
        const float validationLoss = computeValidationLoss();
        model.updateWeightsRPROP(trainingSet.size());

        std::cout << "Epoch " << epoch
                  << "  Training Loss: "       << trainingLoss
                  << "  Validation Loss: "   << validationLoss << "\n";

        if (validationLoss < bestValidationLoss) {
            bestValidationLoss       = validationLoss;
            bestEpoch                = epoch;
            epochsWithoutImprovement = 0;
            appendLayerSnapshot(bestWeights, bestBiases, layers);
        } else {
            ++epochsWithoutImprovement;
            if (epochsWithoutImprovement >= patience) {
                std::cout << "Early Stopping at Epoch " << epoch + 1
                          << " (no improvement for " << patience << " epochs)\n";
                break;
            }
        }
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].W() = bestWeights[i];
        layers[i].b() = bestBiases[i];
    }
    

    std::cout << "Training finished! Best Epoch: " << bestEpoch
              << "  Best Validation Loss: " << bestValidationLoss << "\n";

    float reconstructionError = computeReconstructionError();

    std::cout << "Reconstruction Error: " << reconstructionError << "\n";
}