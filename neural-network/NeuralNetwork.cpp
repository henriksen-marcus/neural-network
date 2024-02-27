#include "NeuralNetwork.h"

#include <cassert>
#include <iostream>

NeuralNetwork::NeuralNetwork(const NNConstructionInfo& constructionInfo)
{
    networkLayers.reserve(constructionInfo.topology.size());
    for (size_t i = 0; i < constructionInfo.topology.size(); i++)
    {
        // Input layer shouldn't have any weights, so set numInputs to 0
        networkLayers.emplace_back(
            constructionInfo.topology[i],
            i == 0 ? 0 : constructionInfo.topology[i - 1].numNeurons);
    }
}

std::vector<double> NeuralNetwork::forwardPropagate(const std::vector<double>& input)
{
    // Input size does not match the number of inputs for the network
    assert(input.size() == networkLayers[0].neurons.size());
    
    // The input layer is just there as a container for the input data, we don't need
    // to calculate any output for it (just take it directly)

    // Initialize the input layer with the input data
    for (size_t i = 0; i < input.size(); i++)
        networkLayers[0].neurons[i].output = input[i];
    
    // Forward propagate
    for (size_t i = 1; i < networkLayers.size(); i++) // Skip input layer
    {
        for (auto& neuron : networkLayers[i].neurons)
            neuron.feedForward(networkLayers[i - 1].neurons); // Send the output from the previous layer
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

    const double meanSquareError = errorSum/(double)outputLayer.neurons.size();
    
    // Calculate output layer gradients (different function for output layer)
    for (size_t i = 0; i < outputLayer.neurons.size() - 1; i++)
    {
        outputLayer.neurons[i].calculateOutputGradient(targetOutput[i]);
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
    
    // Update output layer weights and biases
    for (auto& neuron : outputLayer.neurons)
    {
        // Send the neurons from the previous layer
        neuron.updateWeights(networkLayers[networkLayers.size() - 2].neurons, true);
        neuron.updateBias();
    }

    // Update weights and biases for hidden layers
    for (size_t i = networkLayers.size() - 2; i > 0; i--)
    {
        NetworkLayer& layer = networkLayers[i];
        
        for (auto& neuron : layer.neurons)
        {
            // Send the neurons from the previous layer
            neuron.updateWeights(networkLayers[i-1].neurons, false);
            neuron.updateBias();
        }
    }
    
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
