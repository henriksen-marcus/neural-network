#pragma once

#include <vector>
#include "Neuron.h"
#include "NNConstructionInfo.h"

/**
 * \brief A layer in the neural network, containing a number of neurons.
 */
struct NetworkLayer
{
    NetworkLayer(const LayerInfo& layerInfo, size_t numNeuronInputs)
    {
        neurons.reserve(layerInfo.numNeurons);
        for (size_t i = 0; i < layerInfo.numNeurons; i++)
        {
            neurons.emplace_back(numNeuronInputs, layerInfo.learningRate, layerInfo.activationFunction);
        }
    }
    
    std::vector<Neuron> neurons;
};
