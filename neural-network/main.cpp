/*
 * https://github.com/henriksen-marcus/neural-network
 */

#include <iostream>
#include <conio.h>
#include "NeuralNetwork.h"
#include "examples/ExampleImageRecognition.h"
#include "examples/ExampleXOR.h"





int main()
{
    std::cout << "Neural Network\n";

    /*ExampleXOR exampleXOR;
    exampleXOR.Start();*/

    ExampleImageRecognition exampleImageRecognition;
    exampleImageRecognition.Start();
    
    return 0;
}
