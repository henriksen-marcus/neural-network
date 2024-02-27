#include "Neuron.h"

#include <iostream>

#include "NetworkLayer.h"

Neuron::Neuron(int numInputs, double* learningRate)
{
    std::random_device randomDevice;
    std::mt19937 randomNumberGenerator(randomDevice());

    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    weights.reserve(numInputs);
    for (int i = 0; i < numInputs; i++)
    {
        // Initialize weights with random values between -1 and 1
        weights.push_back(distribution(randomNumberGenerator));
    }

    inputs.reserve(numInputs);
    bias = distribution(randomNumberGenerator);
    originalOutput = 0.0;
    this->learningRate = learningRate;
}

void Neuron::calculateOutputGradient(double targetOutput)
{
    // Expected output - predicted output
    errorDelta = targetOutput - output;
    errorGradient = errorDelta * activateDerivative(output);
}

void Neuron::calculateHiddenGradient(const NetworkLayer& layerToTheRight, size_t index)
{
    // Sum up the error for each neuron in the next layer
    double sum = 0.0;
    for (size_t i = 0; i < layerToTheRight.neurons.size() - 1; i++)
    {
        const Neuron& neuron = layerToTheRight.neurons[i];
        sum += neuron.weights[index] * neuron.errorGradient;
    }
    
    errorGradient = sum * activateDerivative(output);
}

double Neuron::activate(double input)
{
    return 1 / (1 + exp(-input)); // Sigmoid activation function
}

double Neuron::activateDerivative(double input)
{
    return input * (1 - input); // Derivative of the sigmoid activation function
}

void Neuron::feedForward(const std::vector<Neuron>& neuronsOfPreviousLayer)
{
    inputs.clear();
    //std::cout << "Neuron::feedForward\n";
    double sum = 0.0;
    for (size_t i = 0; i < neuronsOfPreviousLayer.size(); i++)
    {
        //std::cout << "   Neuron::feedForward: i: " << i << " Neurons of previous layer size: " << neuronsOfPreviousLayer.size() << "\n";
        inputs.push_back(neuronsOfPreviousLayer[i].output);
        
        // Multiply each input by this neuron's corresponding weight and sum them up
        sum += neuronsOfPreviousLayer[i].output * weights[i];
    }
    sum += bias;
    originalOutput = sum;
    output = activate(sum);

    //std::cout << "   Neuron::feedForward: inputs size: " << inputs.size() << "\n";
}

void Neuron::updateWeights(const std::vector<Neuron>& neuronsOfPreviousLayer, bool isOutputLayer)
{
    //std::cout << "      Neuron::updateWeights: isOutputLayer: " << isOutputLayer << " Weights length: " << weights.size() << " Inputs length: " << inputs.size()  << ".\n";
    
    if (isOutputLayer)
    {
        for (size_t i = 0; i < weights.size(); i++)
        {
            double newWeight = weights[i] + *(learningRate) * inputs[i] * errorDelta;
            //std::cout << "New weight = " << weights[i] << " + " << *learningRate << " * " << inputs[i] << " * " << errorDelta << " = " << newWeight << ".\n";
            if (newWeight != weights[i])
            {
                //std::cout << "      Neuron::updateWeights: New weight: " << newWeight << " Old weight: " << weights[i] << " Learning rate: " << *learningRate << " Inputs: " << inputs[i] << " Error delta: " << errorDelta << ".\n";
            }
            weights[i] = newWeight;
        }
    }
    else
    {
        for (size_t i = 0; i < weights.size(); i++)
        {
            double newWeight = weights[i] + *learningRate * inputs[i] * errorGradient;
            //std::cout << "      Neuron::updateWeights: New weight: " << newWeight << " Old weight: " << weights[i] << " Learning rate: " << *learningRate << " Inputs: " << inputs[i] << " Error delta: " << errorDelta << ".\n";
            weights[i] = newWeight;
        }
    }
}

void Neuron::updateBias()
{
    bias += *learningRate * errorGradient;
}
