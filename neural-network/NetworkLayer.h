#pragma once
#include <vector>
#include "Neuron.h"

/**
 * \brief A layer in the neural network, containing a number of neurons.
 */
struct NetworkLayer
{
    NetworkLayer(int numNeurons, int numNeuronInputs, double* learningRate)
    {
        neurons.reserve(numNeurons);
        for (int i = 0; i < numNeurons; i++)
        {
            neurons.emplace_back(numNeuronInputs, learningRate);
        }
    }
    
    std::vector<Neuron> neurons;
};
