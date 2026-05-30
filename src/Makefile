CXX = g++

DEBUG_FLAGS = -g -fsanitize=address
RELEASE_FLAGS = -fopenmp -O3 -march=native

INCLUDES = -I/usr/include/eigen3 \
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
-I ../Launcher/WeightSerializer

CXXFLAGS = $(RELEASE_FLAGS) $(INCLUDES)

TARGET = main

SRCS = main.cpp \
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


OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

debug:
	$(MAKE) CXXFLAGS="$(DEBUG_FLAGS) $(INCLUDES)"

release:
	$(MAKE) CXXFLAGS="$(RELEASE_FLAGS) $(INCLUDES)"

clean:
	rm -f $(OBJS) $(TARGET)