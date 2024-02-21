#pragma once

#include <random>
#include <vector>

/**
 * @struct Neuron
 * @brief This struct represents a neuron in a neural network.
 *
 * Each neuron has a bias and an output value, as well as an error gradient for backpropagation.
 * The neuron also holds the raw value before applying the activation function to get the output (N).
 * Each neuron has a set of inputs and corresponding weights.
 */
struct Neuron
{
    /**
     * @brief Construct a new Neuron object
     *
     * @param numInputs The number of inputs to the neuron
     * @param randomNumberGenerator A random number generator to initialize the weights and bias
     */
    Neuron(int numInputs, std::mt19937 randomNumberGenerator)
    {
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        
        weights.reserve(numInputs);
        for (int i = 0; i < numInputs; i++)
        {
            // Initialize weights with random values between -1 and 1
            weights.push_back(distribution(randomNumberGenerator));
        }

        bias = distribution(randomNumberGenerator);
        
    }
    ~Neuron() = default;
    
    double bias;
    double output{};

    // Error gradient value for backpropagation
    double errorGradient{};

    // The raw value before applying the activation function to get the output
    double N{};
    
    std::vector<double> inputs;
    std::vector<double> weights;
};
