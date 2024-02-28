#pragma once

class ITrainingExample
{
public:
    virtual ~ITrainingExample() = default;
    virtual void Start() = 0;
};
