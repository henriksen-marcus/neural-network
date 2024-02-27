#include "NeuralNetwork.h"

#include <assert.h>
#include <iostream>
#include <string>

NeuralNetwork::NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer,
                             double learningRate, ActiviationFunction activationFunction)
{
    this->numInputs = numInputs;
    this->numOutputs = numOutputs;
    this->numHiddenLayers = numHiddenLayers;
    this->numNeuronsPerHiddenLayer = numNeuronsPerHiddenLayer;
    this->learningRate = learningRate;

    // Input layer
    networkLayers.emplace_back(numInputs, 1, &learningRate);
    
    for (int i = 0; i < numHiddenLayers; i++)
    {
        // The neurons on "this" layer needs to take in the same amount of inputs as the previous layer has neurons
        networkLayers.emplace_back(numNeuronsPerHiddenLayer, networkLayers.back().neurons.size(), &learningRate);
    }

    // Output layer
    networkLayers.emplace_back(numOutputs, networkLayers.back().neurons.size(), &learningRate);
}

std::vector<double> NeuralNetwork::forwardPropagate(const std::vector<double>& input)
{
    //std::cout << "NeuralNetwork: Start forward propagation.\n";
    
    // Input size does not match the number of inputs for the network
    assert(input.size() == numInputs);
    
    // The input layer is just there as a container for the input data, we don't need
    // to calculate any output for it (just take it directly)

    // Initialize the input layer with the input data
    for (size_t i = 0; i < input.size(); i++)
        networkLayers[0].neurons[i].output = input[i];
    
    // Forward propagate
    for (size_t i = 1; i < networkLayers.size(); i++) // Skip input layer
    {
        //std::cout << "NeuralNetwork: Feedforward layer " << (i == networkLayers.size() - 1 ? "OUTPUT" : std::to_string(i)) << ".\n";
        
        for (auto& neuron : networkLayers[i].neurons)
        {
            neuron.feedForward(networkLayers[i - 1].neurons); // Send the output from the previous layer
        }
    }
    // Forward propagation is done, the output layer now contains the output from the network

    // Return the output layer neurons' output
    std::vector<Neuron> outputLayer = networkLayers.back().neurons;
    
    std::vector<double> output;
    output.reserve(outputLayer.size());

    for (const Neuron& neuron : outputLayer)
        output.emplace_back(neuron.output);
    
    //std::cout << "NeuralNetwork: Forward propagation done.\n";

    // Print neural network vizualization
    /*std::cout << "NeuralNetwork: Neural network visualization:\n";
    for (size_t i = 0; i < networkLayers.size(); i++)
    {
        std::cout << "Layer " << i << ":\n";
        for (size_t k = 0; k < networkLayers[i].neurons.size(); k++)
        {
            std::cout << "  Neuron " << k << ":\n";
            std::cout << "    Output: " << networkLayers[i].neurons[k].output << "\n";
            std::cout << "    Bias: " << networkLayers[i].neurons[k].bias << "\n";
            std::cout << "    ErrGrad: " << networkLayers[i].neurons[k].errorGradient << "\n";
            std::cout << "    Weights:\n";
            for (size_t j = 0; j < networkLayers[i].neurons[k].weights.size(); j++)
            {
                std::cout << "      " << networkLayers[i].neurons[k].weights[j] << "\n";
            }
            if (i == networkLayers.size() - 1)
            {
                std::cout << "    Error difference: " << networkLayers[i].neurons[k].errorDelta << "\n";
            }
        }
    }*/
    
    return output;
}

double NeuralNetwork::backPropagate(const std::vector<double>& input, const std::vector<double>& targetOutput)
{
    //std::cout << "NeuralNetwork: Start backpropagation.\n";
    
    // Calculate overall error (MSE - mean squared error)
    NetworkLayer& outputLayer = networkLayers.back();
    double errorSum = 0.0;

    // Sum up the error for each neuron in the output layer
    for (size_t i = 0; i < outputLayer.neurons.size(); i++)
    {
        outputLayer.neurons[i].errorDelta = targetOutput[i] - outputLayer.neurons[i].output;
        double neronDeltaError = targetOutput[i] - outputLayer.neurons[i].output;
        errorSum += neronDeltaError * neronDeltaError;
    }

    double meanSquareError = errorSum/(double)outputLayer.neurons.size();

    //std::cout << "NeuralNetwork: Mean square error: " << meanSquareError << ".\n";

    //recentAverageError = (recentAverageError * recentAverageSmoothingFactor + meanSquareError) / (recentAverageSmoothingFactor + 1.0);

   // std::cout << "NeuralNetwork: Calculating error gradients.\n";

    //std::cout << "Error difference: " << outputLayer.neurons[0].errorDelta << "\n";
    //std::cout << "Error difference manual: " << targetOutput[0] - outputLayer.neurons[0].output << "\n";
    
    // Calculate output layer gradients (different function for output layer)
    for (size_t i = 0; i < outputLayer.neurons.size() - 1; i++)
    {
        outputLayer.neurons[i].calculateOutputGradient(targetOutput[i]);
        //std::cout << "Output layer gradient: " << outputLayer.neurons[i].errorGradient << "\n";
    }

    // Calculate hidden layer gradients
    for (size_t i = networkLayers.size() - 2; i > 0; i--)
    {
        NetworkLayer& thisLayer = networkLayers[i];
        NetworkLayer& layerToTheRight = networkLayers[i + 1];

        for (size_t k = 0; k < thisLayer.neurons.size(); k++)
        {
            thisLayer.neurons[k].calculateHiddenGradient(layerToTheRight, k);
        }
    }

    // All error gradients have been calculated, now we need to update the weights and biases
    
    //std::cout << "  NeuralNetwork: Update weights and biases for OUTPUT layer.\n";
    int u = 0;
    // Update output layer weights and biases
    for (auto& neuron : outputLayer.neurons)
    {
        //std::cout << "   Neuron " << u++ << ":\n";
        neuron.updateWeights(TODO, true);
        neuron.updateBias();
    }

    // Update weights and biases for hidden layers
    for (size_t i = networkLayers.size() - 2; i > 0; i--)
    {
        //std::cout << "  NeuralNetwork: Update weights and biases for layer " << i << ".\n";
        
        NetworkLayer& layer = networkLayers[i];
        //NetworkLayer& layerToTheRight = networkLayers[i + 1];
        u = 0;
        for (auto& neuron : layer.neurons)
        {
            //std::cout << "   Neuron " << u++ << ":\n";
            neuron.updateWeights(TODO, false);
            neuron.updateBias();
        }
    }

    //std::cout << "NeuralNetwork: Updated weights and biases.\n";
    return meanSquareError;
}

double NeuralNetwork::train(const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& targetOutput)
{
    double MSE = 0.0;
    for (size_t i = 0; i < trainingData.size(); i++)
    {
        std::vector<double> input = trainingData[i];
        std::vector<double> output = forwardPropagate(input);
        MSE = backPropagate(input, targetOutput[i]);
    }
    
    return MSE;
}
