#pragma once

#include "Dataset.hpp"

/**
 * High-level, stateless training pipelines.
 *
 * Design goals:
 *  - Each function is self-contained: it creates its own model/trainer
 *    objects, performs I/O, and prints progress.
 *  - Every function is thread-safe as long as callers use distinct values
 *    of h (which maps to distinct file paths via ModelPaths).
 *  - No global or shared mutable state is touched.
 *
 * The Dataset is passed by const-reference; it is only read, never modified,
 * so it can be shared across OpenMP threads without synchronisation.
 */
namespace TrainingPipeline {

    /**
     * Trains an Autoencoder with hidden size h, then saves its weights to
     *   weights_autoencoder_h<h>.dat
     */
    void trainAutoencoder(int h, const Dataset& dataset);

    /**
     * Pre-condition: weights_autoencoder_h<h>.dat must exist and be non-empty.
     * Loads the Autoencoder, encodes the whole dataset, trains a FeedForward
     * classifier on the encoded representations, then saves its weights to
     *   weights_feedforward_h<h>.dat
     * Throws std::runtime_error if the pre-condition is not met.
     */
    void trainFeedforward(int h, const Dataset& dataset);

    /**
     * Convenience wrapper: runs trainAutoencoder then trainFeedforward in order.
     */
    void trainFull(int h, const Dataset& dataset);

    /**
     * Pre-condition: both weight files for h must exist and be non-empty.
     * Loads both models, runs the test set through the full pipeline, and
     * writes accuracy to
     *   results_test_h<h>.txt
     * Throws std::runtime_error if either pre-condition is not met.
     */
    void runTest(int h, const Dataset& dataset);

} // namespace TrainingPipeline
