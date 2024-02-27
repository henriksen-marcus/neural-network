/*
 * https://github.com/henriksen-marcus/neural-network
 */

#include <iostream>
#include "NeuralNetwork.h"
#include <conio.h>


int main()
{
    std::cout << "Neural Network\n";
    
    // Training data, XOR
    std::vector<std::vector<double>> trainingData = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<double>> targetOutput = {
        {0},
        {1},
        {1},
        {0}
    };

    // Construct the neural network
    NNConstructionInfo nnInfo(2, LayerInfo(1, 0.1, Sigmoid));
    nnInfo.addHiddenLayer(LayerInfo(300, 0.1, Tanh));
    NeuralNetwork nn(nnInfo);
    
    // Train for multiple epochs
    double mse = 0;
    int numEpochs = 10000;
    for (int i = 0; i < numEpochs; i++)
    {
        // Train the network with the entire dataset each epoch
        mse = nn.train(trainingData, targetOutput);
        //std::cout << "============== MSE: " << nn.train(trainingData, targetOutput) << " ==============\n";
    }

    std::cout << "Final MSE: " << mse << "\n";
    std::cout << "Prediction for 0, 0: " << nn.forwardPropagate(trainingData[0])[0] << "\n";
    std::cout << "Prediction for 0, 1: " << nn.forwardPropagate(trainingData[1])[0] << "\n";
    std::cout << "Prediction for 1, 0: " << nn.forwardPropagate(trainingData[2])[0] << "\n";
    std::cout << "Prediction for 1, 1: " << nn.forwardPropagate(trainingData[3])[0] << "\n";
    
    return 0;
}
