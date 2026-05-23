CXX = g++
CXXFLAGS = -fsanitize=address -I/usr/include/eigen3 \
-I Dataset -I MNISTReader -I printTest -I Layer \
-I Layer/Activation -I Layer/Activation/ActivationFunctions \
-I NeuralNetwork -I NeuralNetwork/Autoencoder \
-I Trainer -I Trainer/LossFunction -I Trainer/LossFunction/LossFunctions

TARGET = mnist_viewer

SRCS = main.cpp \
Dataset/Dataset.cpp \
MNISTReader/MNISTReader.cpp \
printTest/printTest.cpp \
Layer/Layer.cpp \
Layer/Activation/ActivationFunctions/ActivationFunctions.cpp \
NeuralNetwork/Autoencoder/Autoencoder.cpp \
Trainer/Trainer.cpp \
Trainer/LossFunction/LossFunctions/LossFunctions.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
