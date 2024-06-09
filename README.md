# Deeplearning Library

## Description
    This library useses CUDA and C++ to implement basic functionality of Deeplearning
    With the use of this library
    1. A sequential model can be created.
    2. Layers can be added to the model
    3. Model compilation
    4. Model training




## Model: Partially functional
    At this stage only sequential model can be used.




## Layers
    Location:
    include: include/Layers.h
    source code: lib/Layers/*

### Dense Layer: Fully functional
    Forward and backward functionality is implemented    
### Batch Normalization Layer: Not functional
    Forward propagation functionality is implemented. Backward propagation is not implemented yet.





## Activation Funcations
    Location:
    include: include/Activations.h
    source code: lib/Activations/*

### Relu, Sigmoid, linear, softman: Funtional




## Optimizer Type: Functional
    Location:
    include: include/Optimizers_type.h
    source code: lib/Optimizers/*

### SGD (Stocastic Gradient Descent): Functional
### SGD_momentum: Functional
### Adagrad: Functional
### Adadelta: Functional
### RMSprop: Functional
### ADAM: Functional




## Losses: Partially Functional
    Location:
    include: include/Losses.h
    source code: lib/Losses/*

### Linear, Mean_Squared_Error, Categorical_Cross_Entropy, Binary_Cross_Entropy: Funcational





## Bunch info:
    master: baseline function working properly
