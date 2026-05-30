#!/bin/bash

# Compile all source files with include paths for each module subdirectory
g++ -O3 -march=native \
    -I /usr/include/eigen3 \
    -I Dataset \
    -I MNISTReader \
    -I Layer \
    -I Layer/Activation \
    -I Layer/Activation/ActivationFunctions \
    -I NeuralNetwork \
    -I NeuralNetwork/Autoencoder \
    -I NeuralNetwork/FeedForward \
    -I Trainer \
    -I Trainer/LossFunction \
    -I Trainer/LossFunction/LossFunctions \
    -I ../Launcher/ModelPaths \
    -I ../Launcher/TrainingPipeline \
    -I ../Launcher/WeightSerializer \
    main.cpp \
    Dataset/Dataset.cpp \
    MNISTReader/MNISTReader.cpp \
    Layer/Layer.cpp \
    Layer/Activation/ActivationFunctions/ActivationFunctions.cpp \
    NeuralNetwork/Autoencoder/Autoencoder.cpp \
    NeuralNetwork/FeedForward/FeedForward.cpp \
    Trainer/Trainer.cpp \
    Trainer/LossFunction/LossFunctions/LossFunctions.cpp \
    ../Launcher/TrainingPipeline/TrainingPipeline.cpp \
    ../Launcher/WeightSerializer/WeightSerializer.cpp 
