#pragma once

#include <vector>
#include "ActivationFunction.h"

struct LayerInfo
{
    size_t numNeurons;
    double learningRate;
    ActiviationFunction activationFunction;

    /**
     * \brief 
     * \param numNeurons The number of neurons in the layer. For the input layer, this is
     * the number of inputs. For the output layer, number of outputs.
     * \param learningRate The rate of change for the weights and biases during training.
     * \param activationFunction Which activation function to use for the neurons in the layer.
     */
    LayerInfo(size_t numNeurons = 0, double learningRate = 0.05, ActiviationFunction activationFunction = ActiviationFunction::Sigmoid)
        : numNeurons(numNeurons), learningRate(learningRate), activationFunction(activationFunction)
    {
    }
};

/**
 * \brief Construction info for a neural network.
 */
struct NNConstructionInfo
{
    std::vector<LayerInfo> topology;

    /**
     * \param inputLayerNumNeurons How many inputs the network should have.
     * We only pass in a number here because the input layer doesn't have any weights or biases.
     * \param outputLayer 
     */
    NNConstructionInfo(size_t inputLayerNumNeurons, const LayerInfo& outputLayer)
    {
        topology.emplace_back(inputLayerNumNeurons);
        topology.push_back(outputLayer);
    }

    /**
     * \brief Insert a new hidden layer into the topology
     * \param layerInfo Info for the layer to be constructed
     */
    void addHiddenLayer(const LayerInfo& layerInfo)
    {
        // Insert right before the output layer
        topology.insert(std::prev(topology.end()), layerInfo);
    }
};
