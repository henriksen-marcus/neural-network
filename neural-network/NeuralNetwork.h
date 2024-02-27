#pragma once
#include <vector>

#include "NetworkLayer.h"

class NeuralNetwork
{
public:
    enum ActiviationFunction
    {
        Sigmoid,
        ReLU
    };

    /**
     * \brief 
     * \param numInputs How many neurons there should be on the first layer (input layer).
     * For image data this would be the amount of pixels in the image
     * \param numOutputs How many neurons there should be on the last layer (output layer)
     * \param numHiddenLayers How many hidden layers in between there should be
     * \param numNeuronsPerHiddenLayer How many neurons each hidden layer should contain
     * \param learningRate The learning rate for the network
     * \param activationFunction The activation function to use for the network 
     */
    NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, double learningRate, ActiviationFunction activationFunction);
    ~NeuralNetwork() = default;

    std::vector<double> forwardPropagate(const std::vector<double>& input);
    double backwardPropagate(const std::vector<double>& input, const std::vector<double>& targetOutput);
    /**
     * \brief Train by forward propagating and backpropagating the network as many times as there are inputs
     * in the trainingData vector.
     * \param trainingData An vector containing a list of input data. Each input is a vector of input values.
     * If you only have one input, use a vector with only one element.
     * \param targetOutput The expected output for each input in the trainingData vector.
     * If you only have one expected output, use a vector with only one element.
     * \return 
     */
    double train(const std::vector<std::vector<double>>& trainingData, const std::vector<std::vector<double>>& targetOutput);

    double predict(const std::vector<double>& input);
    
    int numInputs;
    int numOutputs;
    int numHiddenLayers;
    int numNeuronsPerHiddenLayer;
    double learningRate;
    double error;
    double recentAverageError;
    double recentAverageSmoothingFactor;
    std::vector<NetworkLayer> networkLayers;
};
