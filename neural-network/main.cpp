/*
 * https://github.com/henriksen-marcus/neural-network
 */

#include <iostream>
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
    
    NeuralNetwork nn = NeuralNetwork(2, 1, 2, 100, 0.1, Sigmoid);

    // Train for multiple epochs
    double mse = 0;
    int numEpochs = 10000;
    for (int i = 0; i < numEpochs; i++)
    {
        // Train the network with the entire dataset each epoch
        //nn.train(trainingData, targetOutput);
        /*nn.forwardPropagate(trainingData[1]);
        nn.backwardPropagate(trainingData[1], targetOutput[1]);*/
        mse = nn.train(trainingData, targetOutput);
        //std::cout << "============== MSE: " << nn.train(trainingData, targetOutput) << " ==============\n";
        //_getch();
    }

    std::cout << "MSE: " << mse << "\n";

    //double MSE = 0;
    // Test the network on a single data point
    //auto result = nn.forwardPropagate(trainingData[0]);

    //std::vector<double> result = nn.forwardPropagate(trainingData[0]);
    //std::cout << "Result size: " << result.size() << "\n";
    
    std::cout << "Prediction for 0, 0: " << nn.forwardPropagate(trainingData[0])[0] << std::endl;
    std::cout << "Prediction for 0, 1: " << nn.forwardPropagate(trainingData[1])[0] << std::endl;
    std::cout << "Prediction for 1, 0: " << nn.forwardPropagate(trainingData[2])[0] << std::endl;
    std::cout << "Prediction for 1, 1: " << nn.forwardPropagate(trainingData[3])[0] << std::endl;
    
    //std::cout << "MSE: " << MSE << std::endl;
    //std::cout << "Prediction for 0, 0: " << nn.predict({0,0}) << std::endl;
    return 0;
}
