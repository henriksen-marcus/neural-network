#pragma once
#include <iomanip>
#include <iostream>
#include <vector>

#include "ITrainingExample.h"
#include "../NeuralNetwork.h"
#include "../Timer.h"
#include "vendor/termcolor.hpp"
#include "vendor/mnist_sdk/mnist_reader.hpp"

class ExampleImageRecognition : public ITrainingExample
{
public:
    void Start() override
    {
        run();
    }

protected:
typedef mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> MNIST;
    
    void run() const
    {
        mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

        // Settings for input & output layer
        NNConstructionInfo nnInfo(IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE, LayerInfo(10, 0.08, Sigmoid));

        // Hidden layers, num neurons usually 2x input layer
        nnInfo.addHiddenLayer(LayerInfo(IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE * 2, 0.08, Sigmoid));
        nnInfo.addHiddenLayer(LayerInfo(IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE * 2, 0.08, Sigmoid));
        nnInfo.addHiddenLayer(LayerInfo(IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE, 0.08, Sigmoid));

        NeuralNetwork nn(nnInfo);
        Timer timer;
        timer.Start();

        // For each image in the training set
        for (size_t i = 0; i < 10000/*dataset.training_images.size()*/; i++)
        {
            // For some reason this is much faster then reserving the size
            std::vector<double> inputs(IMAGE_PIXEL_SIZE*IMAGE_PIXEL_SIZE, 0);
            std::vector<double> outputs(10, 0);

            // Prepare the input to the neural network
            for (size_t k = 0; k < IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE; k++)
            {
                // Normalize the value for each pixel to between 0 - 1
                double pixelValue = (unsigned)(dataset.training_images[i][k]) / 255.0;
                inputs[k] = pixelValue;
            }

            /* Find the actual correct number from the training labels,
             * and set the corresponding index in the outputs vector to 1. */
            outputs[dataset.training_labels[i]] = 1;

            // Train the network
            nn.forwardPropagate(inputs);
            double MSE = nn.backPropagate(inputs, outputs);

            if (i % 10 == 0) std::cout << "Trained on " << i << " images. MSE: " << MSE << "\n";
        }
        
        std::cout << "Training took " << timer.Stop() << " seconds.\n";
        
        testResult(nn, dataset, 200);
        testResult(nn, dataset, 5789);
        testResult(nn, dataset, 2);
        testResult(nn, dataset, 377);
    }
    
    void runVerbose() const
    {
        mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

        std::cout << "Number of training images: " << dataset.training_images.size() << "\n";

        // Settings for input & output layer
        NNConstructionInfo nnInfo(IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE, LayerInfo(10, 0.08, Sigmoid));

        // Hidden layers, num neurons usually 2x input layer
        nnInfo.addHiddenLayer(LayerInfo(IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE, 0.08, Sigmoid));
        nnInfo.addHiddenLayer(LayerInfo(IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE, 0.08, Sigmoid));
        nnInfo.addHiddenLayer(LayerInfo(IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE, 0.08, Sigmoid));

        NeuralNetwork nn(nnInfo);
        Timer timer;
        timer.Start();

        // For each image in the training set
        for (size_t i = 0; i < 10/*dataset.training_images.size()*/; i++)
        {
            // For some reason this is much faster then reserving the size
            std::vector<double> inputs(IMAGE_PIXEL_SIZE*IMAGE_PIXEL_SIZE, 0);
            std::vector<double> outputs(10, 0);

            // Prepare the input to the neural network
            for (size_t k = 0; k < IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE; k++)
            {
                // Normalize the value for each pixel to between 0 - 1
                double pixelValue = (unsigned)(dataset.training_images[i][k]) / 255.0;

                // Print the image
                if (pixelValue > 0) std::cout << termcolor::red << 1 << " ";
                else std::cout << 0 << " ";
    
                std::cout << termcolor::reset;
    
                // New line every 28 pixels
                if ((k + 1) % 28 == 0)
                    std::cout << "\n";
                inputs[k] = pixelValue;
            }

            /* Find the actual correct number from the training labels,
             * and set the corresponding index in the outputs vector to 1. */
            outputs[dataset.training_labels[i]] = 1;

            // Print outputs
            std::cout << "Outputs:\n";
            for (int j = 0; j < 10; j++)
                std::cout << j << " ";
            
            std::cout << "\n";
            
            for (auto& o : outputs)
                std::cout << o << " ";
            
            std::cout << "\n" << "Correct number: " << (unsigned)dataset.training_labels[i] << "\n";
            std::cout << "=== Press to continue ===\n";
            _getch();

            // Train the network
            nn.forwardPropagate(inputs);
            double MSE = nn.backPropagate(inputs, outputs);

            if (i % 10 == 0) std::cout << "Trained on " << i << " images. MSE: " << MSE << "\n";
        }
        
        std::cout << "Training took " << timer.Stop() << " seconds.\n";
    }
    
    std::vector<double> loadImage(const MNIST& dataset, size_t index) const
    {
        std::vector<double> inputs;
        inputs.reserve(IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE);

        for (size_t k = 0; k < IMAGE_PIXEL_SIZE * IMAGE_PIXEL_SIZE; k++)
        {
            // Normalize the value for each pixel to between 0 - 1
            double pixelValue = (unsigned)(dataset.training_images.at(index).at(k)) / 255.0;
            inputs.push_back(pixelValue);
        }

        return inputs;
    }
    
    void testResult(NeuralNetwork& nn, const MNIST& dataset, size_t index) const
    {
        std::cout << "Attempting to predict number " << (int)dataset.training_labels[index] << ":\n";
        const auto results = nn.predict(loadImage(dataset, index));

        for (int i = 0; i < 10; i++)
            std::cout << i << "    ";
        std::cout << "\n";
        
        for (const auto res : results)
            std::cout << std::fixed << std::setprecision(2) << res << " ";
        std::cout << "\n";
        
        double highestValue = 0;
        size_t bestIndex = 0;
        for (size_t i = 0; i < results.size(); i++)
        {
            if (results[i] > highestValue)
            {
                highestValue = results[i];
                bestIndex = i;
            }
        }

        std::cout << "The network predicted the number " << bestIndex << " with a confidence of " << highestValue*100 << "%.\n";
    }
    
    const std::string MNIST_DATA_LOCATION = "vendor/_mnist_dataset";
    const size_t IMAGE_PIXEL_SIZE = 28;
};
