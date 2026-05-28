CXX = g++

DEBUG_FLAGS = -g -fsanitize=address
RELEASE_FLAGS = -O3 -march=native

INCLUDES = -I/usr/include/eigen3 \
-I Dataset \
-I MNISTReader \
-I printTest \
-I Layer \
-I Layer/Activation \
-I Layer/Activation/ActivationFunctions \
-I NeuralNetwork \
-I NeuralNetwork/Autoencoder \
-I NeuralNetwork/FeedForward \
-I Trainer \
-I Trainer/LossFunction \
-I Trainer/LossFunction/LossFunctions \

CXXFLAGS = $(RELEASE_FLAGS) $(INCLUDES)

TARGET = mnist_viewer

SRCS = main.cpp \
Dataset/Dataset.cpp \
MNISTReader/MNISTReader.cpp \
printTest/printTest.cpp \
Layer/Layer.cpp \
Layer/Activation/ActivationFunctions/ActivationFunctions.cpp \
NeuralNetwork/Autoencoder/Autoencoder.cpp \
NeuralNetwork/FeedForward/FeedForward.cpp \
Trainer/Trainer.cpp \
Trainer/LossFunction/LossFunctions/LossFunctions.cpp \

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