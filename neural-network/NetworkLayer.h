﻿#pragma once
#include <vector>
#include "Neuron.h"

struct NetworkLayer
{
    NetworkLayer(int numNeurons, int numNeuronInputs)
    {
        std::random_device randomDevice;
        std::mt19937 randomNumberGenerator(randomDevice());
        
        neurons.reserve(numNeurons);
        for (int i = 0; i < numNeurons; i++)
        {
            neurons.emplace_back(numNeuronInputs, randomNumberGenerator);
        }
    }
    ~NetworkLayer() = default;
    
    std::vector<Neuron> neurons;
};