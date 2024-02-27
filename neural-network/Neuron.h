#pragma once

#include <random>
#include <vector>
//#include "NetworkLayer.h"

enum ActiviationFunction : int;
struct NetworkLayer;

/**
 * @struct Neuron
 * @brief This struct represents a neuron in a neural network.
 * Each neuron has a bias and an output value, as well as an error gradient for backpropagation.
 * The neuron also holds the raw value before applying the activation function to get the output (N).
 * Each neuron has a set of inputs and corresponding weights.
 */
class Neuron
{
public:
    /**
     * \param numInputs The number of inputs the neuron should be able to handle
     * \param learningRate A pointer to the learning rate stored in each layer
     */
    Neuron(size_t numInputs, double learningRate, ActiviationFunction activationFunction);

    /**
     * \brief 
     * \param targetOutput 
     */
    void calculateOutputGradient(double targetOutput);
    void calculateHiddenGradient(const NetworkLayer& layerToTheRight, size_t index);


    /**
     * \brief Calculates the activation amount for the neuron.
     * \param input The summed input to the neuron.
     * \return The activation amount.
     */
    static double activate(double input);

    /**
     * \brief 
     * \param input 
     * \return 
     */
    static double activateDerivative(double input);

    /**
     * \brief Processes the output from the previous layer and calculates the output for this neuron
     * \param neuronsOfPreviousLayer The neurons from the previous layer
     */
    void feedForward(const std::vector<Neuron>& neuronsOfPreviousLayer);
    void updateWeights(const std::vector<Neuron>& neuronsOfPreviousLayer, bool isOutputLayer);
    void updateBias();

    // The raw output value of the neuron before applying the activation function
    double originalOutput;

    // The activated, predicted output value
    double output{};

    // Error gradient value for backpropagation
    double errorGradient{};

    /**
     * \brief aka. Error difference.
     * The diff between the expected output and the predicted output.
     * Only used for output layer neurons.
     */
    double errorDelta{};

    std::vector<double> weights;

protected:
    double bias;
    // Learning rate from the parent layer
    double learningRate;
    ActiviationFunction activationFunction;
};
