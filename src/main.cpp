#include "Dataset.hpp"
#include "TrainingPipeline.hpp"
#include "ModelPaths.hpp"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <limits>       // std::numeric_limits

#ifdef _OPENMP
  #include <omp.h>
#endif

// ── Constants ─────────────────────────────────────────────────────────────────

static const std::vector<int> ALL_H_VALUES = {25, 50, 75, 100, 125};

// Sentinel value returned by selectH() when the user requests full parallel run
static constexpr int H_PARALLEL = 6;

// ── UI helpers ────────────────────────────────────────────────────────────────

/**
 * Flushes any leftover newline in the input buffer, then reads one integer.
 * Retries until the input is a valid integer.
 */
static int readInt(const std::string& prompt)
{
    int value = 0;
    while (true) {
        std::cout << prompt;
        if (std::cin >> value) break;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cerr << "  Input non valido. Riprova.\n";
    }
    return value;
}

/**
 * Asks the user to choose a hidden-layer size h.
 *
 * Returns:
 *   1-5  → index into ALL_H_VALUES  (use ALL_H_VALUES[choice - 1] to get h)
 *   6    → run all five values in parallel (H_PARALLEL sentinel)
 */
static int selectH()
{
    std::cout << "\n"
              << "┌─────────────────────────────────────────────────┐\n"
              << "│        Selezione dimensione Hidden Layer (h)     │\n"
              << "├─────────────────────────────────────────────────┤\n"
              << "│  1.  h = 25                                      │\n"
              << "│  2.  h = 50                                      │\n"
              << "│  3.  h = 75                                      │\n"
              << "│  4.  h = 100                                     │\n"
              << "│  5.  h = 125                                     │\n"
              << "│  6.  Addestramento Parallelo (tutti – OpenMP)    │\n"
              << "└─────────────────────────────────────────────────┘\n";

    int choice = 0;
    while (choice < 1 || choice > 6)
        choice = readInt("Scelta [1-6]: ");

    if (choice == H_PARALLEL) {
#ifndef _OPENMP
        std::cout << "  ⚠  OpenMP non disponibile: l'esecuzione sarà sequenziale.\n";
#else
        std::cout << "  → Modalità parallela: " << omp_get_max_threads()
                  << " thread disponibili.\n";
#endif
    } else {
        std::cout << "  → h = " << ALL_H_VALUES[choice - 1] << " selezionato.\n";
    }

    return choice;
}

/**
 * Displays the main menu and returns the chosen option (0-4).
 */
static int showMenu()
{
    std::cout << "\n"
              << "┌─────────────────────────────────────────────────────────┐\n"
              << "│                     Menu Principale                     │\n"
              << "├─────────────────────────────────────────────────────────┤\n"
              << "│  1.  Addestramento Completo (Autoencoder + FeedForward)  │\n"
              << "│  2.  Addestra solo Autoencoder                           │\n"
              << "│  3.  Addestra solo FeedForward  (richiede pesi AE)       │\n"
              << "│  4.  Test Set + Calcolo Accuracy (richiede pesi AE+FF)   │\n"
              << "│  0.  Esci                                                │\n"
              << "└─────────────────────────────────────────────────────────┘\n";

    int choice = -1;
    while (choice < 0 || choice > 4)
        choice = readInt("Scelta [0-4]: ");

    return choice;
}

// ── Pipeline dispatching ──────────────────────────────────────────────────────

// Function pointer type matching every TrainingPipeline function.
using PipelineFn = void(*)(int, const Dataset&);

/**
 * Executes fn(h, dataset) and catches/prints any exception so that
 * the main loop can continue rather than terminate.
 */
static void runSingle(PipelineFn fn, int h, const Dataset& dataset) noexcept
{
    try {
        fn(h, dataset);
    } catch (const std::exception& e) {
        // Use cerr; in OpenMP context each line is an atomic write on most
        // implementations, so interleaving of complete lines is acceptable.
        std::cerr << "\n[ERRORE – h=" << h << "] " << e.what() << "\n";
    }
}

/**
 * Runs fn for every value in ALL_H_VALUES, in parallel if OpenMP is available.
 *
 * Thread-safety guarantee:
 *   - Dataset is const → safe shared read.
 *   - Each iteration constructs its own Autoencoder/FeedForward/Trainer
 *     objects on the stack → no shared mutable state.
 *   - File paths are derived from h → no two threads touch the same file.
 *   - Layer::Layer uses thread_local mt19937 → safe weight initialisation.
 */
static void runParallel(PipelineFn fn, const Dataset& dataset)
{
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(ALL_H_VALUES.size()); ++i)
        runSingle(fn, ALL_H_VALUES[i], dataset);
#else
    std::cerr << "  ⚠  OpenMP non disponibile. Esecuzione sequenziale.\n";
    for (int h : ALL_H_VALUES)
        runSingle(fn, h, dataset);
#endif
}

/**
 * Maps menuChoice (1-4) to the correct PipelineFn, then dispatches it
 * either for a single h or for all five values in parallel.
 *
 * @param menuChoice  1-4 (0 = exit, handled by the caller)
 * @param hChoice     1-5 for a specific h, or H_PARALLEL (6) for all
 */
static void dispatch(int menuChoice, int hChoice, const Dataset& dataset)
{
    PipelineFn fn = nullptr;
    switch (menuChoice) {
        case 1: fn = TrainingPipeline::trainFull;        break;
        case 2: fn = TrainingPipeline::trainAutoencoder; break;
        case 3: fn = TrainingPipeline::trainFeedforward; break;
        case 4: fn = TrainingPipeline::runTest;          break;
        default:
            std::cerr << "  Opzione menu fuori range.\n";
            return;
    }

    if (hChoice == H_PARALLEL) {
        runParallel(fn, dataset);
    } else {
        runSingle(fn, ALL_H_VALUES[hChoice - 1], dataset);
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main()
{
    std::cout << "╔══════════════════════════════════════════════════╗\n"
              << "║  Neural Network: Autoencoder + FeedForward MNIST ║\n"
              << "╚══════════════════════════════════════════════════╝\n";

    // Load dataset once (expensive I/O + preprocessing); all pipeline calls
    // share this read-only object.
    std::cout << "\nCaricamento dataset MNIST...\n";
    const Dataset dataset;

    bool running = true;
    while (running) {
        const int hChoice   = selectH();
        const int menuChoice = showMenu();

        if (menuChoice == 0) {
            running = false;
        } else {
            dispatch(menuChoice, hChoice, dataset);
        }
    }

    std::cout << "\nProgramma terminato.\n";
    return 0;
}
