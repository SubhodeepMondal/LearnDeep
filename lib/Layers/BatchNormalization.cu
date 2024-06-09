#include <map>
#include "NDynamicArray.h"
#include "MathLibrary.h"
#include "Activations.h"
#include "OptimizationType.h"
#include "Optimizers.h"
#include "Forward_Propagation.h"
#include "Layers.h"

void BatchNormalization::initilizeInput()
{
    Layer_ptr *ptr = in_vertices;
    NDArray<double, 0> input_dimension;
    if (!isInputInitilized)
    {
        if (ptr)
        {
            input_dimension = ptr->layer->getOutput();
            input = NDArray<double, 0>(input_dimension.getNoOfDimensions(), input_dimension.getDimensions(), 0);
            input.initPreinitilizedData(input_dimension.getData());
            isInputInitilized = 1;
        }
    }
}

void BatchNormalization::initilizeOutputIntermediate(unsigned batch_size)
{
    unsigned no_of_features = output.getNoOfDimensions(), i;
    unsigned *a = new unsigned[no_of_features + 1];
    no_of_features++;

    for (i = 0; i < no_of_features - 1; i++)
        a[i] = output.getDimensions()[i];
    a[i] = batch_size;

    target = NDArray<double, 1>(output.getNoOfDimensions(), output.getDimensions(), 0);
    output_gpu = NDArray<double, 1>(no_of_features, a);
    difference = NDArray<double, 1>(no_of_features, a);

    delete[] a;
}

void BatchNormalization::initilizeInputIntermediate(unsigned batch_size)
{
    unsigned dims = input.getNoOfDimensions() + 1;
    unsigned i;
    unsigned *a = new unsigned[dims];

    Layer_ptr *ptr = in_vertices;

    for (i = 0; i < dims - 1; i++)
        a[i] = input.getDimensions()[i];
    a[i] = batch_size;

    input_gpu = NDArray<double, 1>(dims, a, 0);
    if (ptr)
    {
        input_gpu.initPreinitilizedData(ptr->layer->getOutputDataGPU().getData());
    }

    delete[] a;
}

void BatchNormalization::initilizeBetaGammaIntermediate(unsigned batch_size)
{
    gamma_gpu = NDArray<double, 1>(gamma.getNoOfDimensions(), gamma.getDimensions());
    beta_gpu = NDArray<double, 1>(beta.getNoOfDimensions(), beta.getDimensions());

    gamma_gpu.initData(gamma.getData());
    beta_gpu.initData(beta.getData());
}

void BatchNormalization::initilizeBetaGamma()
{
    beta = NDArray<double, 0>(input.getNoOfDimensions(), input.getDimensions());
    gamma = NDArray<double, 0>(input.getNoOfDimensions(), input.getDimensions());

    mean = NDArray<double, 0>(input.getNoOfDimensions(), input.getDimensions());
    std_div = NDArray<double, 0>(input.getNoOfDimensions(), input.getDimensions());
    beta.initRandData(-1, 1);
    gamma.initRandData(-1, 1);
}

void BatchNormalization::initilizeOutput()
{
    output = NDArray<double, 0>(input.getNoOfDimensions(), input.getDimensions());
}

void BatchNormalization::updateStream(cudaStream_t stream)
{
    this->stream = stream;
}

void BatchNormalization::updateOptimizer(std::string optimizer)
{
}

BatchNormalization::BatchNormalization(std::string layer_name = "Batch Normalization") : Layer("batch_normalization")
{
    this->layer_name = layer_name;
    isInputInitilized = 0;
}

BatchNormalization::BatchNormalization(NDArray<unsigned, 0> input_shape, std::string layer_name = "Batch Normalization") : Layer("batch_normalization")
{
    isInputInitilized = 0;
}

void BatchNormalization::layerProperties()
{
    this->layer_name = layer_name;
    std::cout << layer_type << ": " << layer_name;
    input.printDimensions();
    std::cout << "\n";
}

NDArray<double, 0> BatchNormalization::getOutput()
{

    return output;
}

return_parameter BatchNormalization::searchDFS(search_parameter search)
{
    return_parameter return_param;
    Layer_ptr *ptr = out_vertices;

    switch (Flag.search_flags[search.search_param])
    {
    case this->Flag.summary:
    {
        layerProperties();
        break;
    }

    case this->Flag.backward_propagation:
    {
        break;
    }

    case this->Flag.compile:
    {
        initilizeInput();
        initilizeBetaGamma();
        initilizeOutput();
        layer_optimizer = search.String;
        break;
    }
    case this->Flag.prepare_training:
    {
        updateStream(search.cuda_Stream);
        initilizeOutputIntermediate(search.unsigned_index);
        initilizeInputIntermediate(search.unsigned_index);
        initilizeBetaGammaIntermediate(search.unsigned_index);
        updateOptimizer(layer_optimizer);

        break;
    }

    case this->Flag.forward_propagation:
    {
        fit(input_gpu, gamma_gpu, beta_gpu, output_gpu, stream);
        return_param.Data_GPU = output_gpu;
        // return return_param;
        break;
    }

    case this->Flag.initilize_input_intermidiate:
    {
        // initilizeInputGPU(search.double_Ptr);
        break;
    }

    case this->Flag.initilize_output_intermidiate:
    {
        // initilizeTarget(search.double_Ptr);
        break;
    }

    case this->Flag.initilize_difference:
    {
        // difference.initData(search.Data_GPU.getData());
        break;
    }

    case this->Flag.print_parameters:
    {
        printGammaGPU();
        printBetaGPU();
        // printWeightGPU();
        // printBiasesGPU();
        // printWeightes();
        // printBiases();
        break;
    }

    case this->Flag.predict:
    {

        // predict(input, weights, biases, output);
        // activation->activate(output);
        break;
    }

    case this->Flag.commit_weights_biases:
    {
        // commitWeightsBiases();
        break;
    }
    }

    while (ptr)
    {
        return_param = ptr->layer->searchDFS(search);
        ptr = ptr->next;
    }
    return return_param;
}

void BatchNormalization::searchBFS(search_parameter search)
{
    Layer_ptr *ptr = in_vertices;

    switch (Flag.search_flags[search.search_param])
    {
    case this->Flag.backward_propagation:
    {

        // optimizer->optimize(input_gpu, weights_gpu, biases_gpu, delta_activation, difference, delta_input, stream);
        break;
    }
    }

    while (ptr)
        ptr = ptr->next;

    ptr = in_vertices;

    while (ptr)
    {

        ptr->layer->searchBFS(search);
        ptr = ptr->next;
    }
}

void BatchNormalization::printGammaGPU()
{
    std::cout << layer_name << ": Gamma GPU: \n";
    gamma_gpu.printData();
}

void BatchNormalization::printBetaGPU()
{
    std::cout << layer_name << ": Beta GPU: \n";
    beta.printData();
}