
#include <iostream>

#include "NetworkLayer.h"


int main()
{
    std::cout << "Hello World!\n";

    NetworkLayer layer(3, 2);
    std::cout << "Bias " << layer.neurons[0].bias << "\n";
}
