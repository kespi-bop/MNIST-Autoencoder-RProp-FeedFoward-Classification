#!/bin/bash

# Compile all source files with include paths for each module subdirectory
g++ -fsanitize=address \
    -I /usr/include/eigen3 \
    -I Dataset \
    -I MNISTReader \
    -I printTest \
    -I Layer \
    -I Layer/Activation \
    -I Layer/Activation/ActivationFunctions \
    -I NeuralNetwork \
    -I NeuralNetwork/Autoencoder \
    main.cpp \
    printTest/printTest.cpp \
    Dataset/Dataset.cpp \
    MNISTReader/MNISTReader.cpp \
    Layer/Layer.cpp \
    Layer/Activation/ActivationFunctions/ActivationFunctions.cpp \
    NeuralNetwork/Autoencoder/Autoencoder.cpp \
    -o mnist_viewer