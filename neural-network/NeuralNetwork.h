#pragma once

#include <vector>
#include "ActivationFunction.h"
#include "NetworkLayer.h"

struct LayerInfo;
/**
 * \brief A complete neural network with a number of layers, each containing a number of neurons.
 * Has the ability to predict outputs and adjust weights and biases through training.
 */
class NeuralNetwork
{
public:
    NeuralNetwork(const NNConstructionInfo& constructionInfo);

    /**
     * \brief Process the input data through the network. This is the same as
     * using the network to predict the output for the given input data.
     * \param input The input data to process
     * \return A vector containing the output from each output layer neuron in the network
     */
    std::vector<double> forwardPropagate(const std::vector<double>& input);

    /**
     * \brief Calculate the error for the output layer and then backpropagate the error,
     * updating the weights and biases for each neuron in the network.
     * \param input The input data processed in the last forward propagation.
     * \param targetOutput The expected output for the given input data.
     * \return The mean squared error (MSE) from the backpropagation.
     */
    double backPropagate(const std::vector<double>& input, const std::vector<double>& targetOutput);
    
    /**
     * \brief Train by forward propagating and backpropagating the network as many times as there are inputs
     * in the trainingData vector.
     * \param trainingData An vector containing a list of input data. Each input is a vector of input values.
     * If you only have one input, use a vector with only one element.
     * \param targetOutput The expected output for each input in the trainingData vector.
     * If you only have one expected output, use a vector with only one element.
     * \return The mean squared error (MSE) for the last backpropagation
     */
    double train(const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& targetOutput);

    /**
     * \brief Predict the output for a given input. Calls forwardPropagate.
     * \param input The input data to process.
     * \return A vector containing the output from each output layer neuron in the network.
     */
    std::vector<double> predict(const std::vector<double>& input)
    {
        return forwardPropagate(input);
    }

protected:
    std::vector<NetworkLayer> networkLayers;
};
