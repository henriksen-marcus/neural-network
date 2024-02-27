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
     * \brief Calculate the error gradient for this neuron
     * if it's in the the output layer.
     * \param targetOutput The target output for this neuron.
     */
    void calculateOutputGradient(double targetOutput);

    /**
     * \brief Calculate the error gradient for this neuron
     * if it's in a hidden layer. Uses the error gradients
     * of the neurons in the next layer.
     * \param layerToTheRight The next layer in the network (i+1)
     * \param index The index of this neuron in the current layer.
     */
    void calculateHiddenGradient(const NetworkLayer& layerToTheRight, size_t index);

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
    /**
     * \brief Calculates the activation amount for the neuron.
     * \param input The summed input to the neuron.
     * \return The activation amount.
     */
    double activate(double input) const;

    /**
     * \brief Calculates the derivative of the activation function for the neuron.
     * \param input The activated output from this neuron.
     * \return 
     */
    double activateDerivative(double input) const;
 
    double bias;
    // Learning rate from the parent layer
    double learningRate;
    ActiviationFunction activationFunction;
};
