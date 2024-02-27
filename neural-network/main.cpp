
#include <iostream>

#include "NetworkLayer.h"
#include "NeuralNetwork.h"
#include <conio.h>


int main()
{
    std::cout << "Hello World!\n";

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
    
    // Increase the number of neurons in the hidden layer
    NeuralNetwork nn = NeuralNetwork(2, 1, 2, 100, 0.1, NeuralNetwork::Sigmoid);

    // Train for multiple epochs
    int numEpochs = 5000;
    for (int i = 0; i < numEpochs; i++)
    {
        // Train the network with the entire dataset each epoch
        //nn.train(trainingData, targetOutput);
        /*nn.forwardPropagate(trainingData[1]);
        nn.backwardPropagate(trainingData[1], targetOutput[1]);*/
        std::cout << "============== MSE: " << nn.train(trainingData, targetOutput) << " ==============\n";
        //_getch();
    }

    //double MSE = 0;
    // Test the network on a single data point
    //MSE = nn.backwardPropagate(trainingData[0], targetOutput[0]);
    
    //std::cout << "MSE: " << MSE << std::endl;
    //std::cout << "Prediction for 0, 0: " << nn.predict({0,0}) << std::endl;
    return 0;
}
