
#include <map>
#include <iostream>
#include "NDynamicArray.h"
#include "MathLibrary.h"
#include "Activations.h"
#include "Losses.h"
#include "Metrics.h"
#include "OptimizationType.h"
#include "Optimizers.h"
#include "Forward_Propagation.h"
#include "Layers.h"
#include "Model.h"

Model::Model(std::string str) : input(NULL), output(NULL)
{
    switch (st_models.model[str])
    {
    case this->st_models.sequential:
    {
        model_type = str;
        input_layer_count = output_layer_cout = 1;
        break;
    }

    default:
        break;
    }
}

void Model::add(Layer *layers) {}

void Model::getModelType()
{
    std::cout << model_type << "\n";
}

/// @brief Compiles the added layer to the model. allocates memory to the vectors. creates the graph for the model.
/// @tparam None
/// @param loss String value, describes which loss should be considered for optimization (i.e mae, )
/// @param optimizer String value, describes which optimizer to be used
/// @param metrics String value, which metrics to be used for model performance evaluation after each iteration.
/// @return output (call by referance)
void Model::compile(std::string loss, std::string optimizer, std::string metrics) {}

/// @brief Sumarizes model with individual layer description
/// @tparam None
/// @return None
void Model::summary()
{
    Layer *ptr = input;
    getModelType();
    search.search_param = "summary";
    ptr->searchDFS(search);
}

/// @brief Fits the model with respective parameters and hyper parameters.
/// @tparam None
/// @param X Input in NDArray
/// @param Y Target in NDArray
/// @param epochs no of iterations training is going to repeat
/// @param batch_size no of training instances for each epochs
/// @return output (call by referance)
void Model::fit(NDArray<double, 0> X, NDArray<double, 0> Y, int epochs, int batch_size)
{
    unsigned no_of_sample, no_of_input_feature, batch_index, data_range, epoch_range, mod, quotent;
    unsigned a[2];
    Layer *ptr;
    NDMath math;
    std::random_device generator;
    std::uniform_int_distribution<int> distribution;
    NDArray<double, 1> Cost;
    NDArray<double, 1> y_predict, difference, y_target_train, y_predict_train;
    NDArray<double, 1> accuracy(1, 1);

    cudaStream_t stream;
    cudaSetDevice(0);

    no_of_input_feature = X.getDimensions()[0];
    a[0] = Y.getDimensions()[0];
    a[1] = batch_size;
    no_of_sample = X.getDimensions()[1];
    data_range = (unsigned)no_of_sample / 4;
    epoch_range = epochs / 4;

    X_input = NDArray<double, 1>(X.getNoOfDimensions(), no_of_input_feature, data_range);
    y_target = NDArray<double, 1>(Y.getNoOfDimensions(), a[0], data_range);
    difference = NDArray<double, 1>(Y.getNoOfDimensions(), a);
    y_target_train = NDArray<double, 1>(Y.getNoOfDimensions(), a, 0);
    y_predict_train = NDArray<double, 1>(Y.getNoOfDimensions(), a);
    y_predict = y_target;

    Cost = NDArray<double, 1>(1, 1);

    distribution = std::uniform_int_distribution<int>(0, data_range);

    cudaStreamCreate(&stream);

    std::cout << "preparing for Training:";

    ptr = input;
    search.cuda_Stream = stream;
    search.unsigned_index = batch_size;
    search.search_param = "prepare_training";
    ptr->searchDFS(search);

    for (int i = 0; i < epochs; i++)
    {

        unsigned index;
        batch_index = distribution(generator);
        batch_index = (data_range - batch_index) > batch_size ? batch_index : (data_range - batch_size);

        // getting data from RAM to GPU RAM in 4 batches.
        mod = i % epoch_range;
        if (!mod)
        {
            quotent = i / epoch_range;
            X_input.initData(X.getData() + quotent * data_range);
            y_target.initData(Y.getData() + quotent * data_range);
        }

        std::cout << "Epoch: " << i + 1 << ", batch index: " << batch_index + quotent * data_range << " ";


        // In model graph pointing the the X_input to proper index.
        ptr = input;
        index = batch_index * no_of_input_feature;
        search.double_Ptr = X_input.getData() + index;
        search.search_param = "initilize_input_intermidiate";
        ptr->searchDFS(search);


        // In model graph pointing the Y_target to proper index.
        index = batch_index * Y.getDimensions()[0];
        y_target_train.initPreinitilizedData(y_target.getData() + index);

        // Forward propagation for the model
        ptr = input;
        search.search_param = "forward_propagation";
        ret_param = ptr->searchDFS(search);
        y_predict = ret_param.Data_GPU;


        // Finding loss for the model
        ptr = output;
        loss->findLoss(y_predict, y_target_train, difference, Cost, stream);
        math.argMax(y_predict, y_predict_train, stream);
        // metric->accuracy(y_predict_train, y_target_train, accuracy, stream);
        std::cout << "Loss: ";
        Cost.printData();
        // std::cout << "Accuracy: ";
        // accuracy.printData();
        std::cout << "\n";


        // Feeding the loss difference to the model graph
        search.Data_GPU = difference;
        search.search_param = "initilize_difference";
        ptr->searchDFS(search);


        // Back propagation to optimize the network
        search.search_param = "backward_propagation";
        ptr->searchBFS(search);
    }

    // Cost.printData();
    Cost.destroy();

    // Storing the data back to cpu
    ptr = input;
    search.search_param = "commit_weights_biases";
    ptr->searchDFS(search);
}

/// @brief Fits the model with respective parameters and hyper parameters.
/// @tparam None
/// @param X Input in NDArray
/// @param Y Target in NDArray
/// @param epochs no of iterations training is going to repeat
/// @param batch_size no of training instances for each epochs
/// @return output (call by referance)
NDArray<double, 0> Model::predict(NDArray<double, 0> X)
{
    NDArray<double, 0> X_input, y_predict;
    /*
    unsigned i, no_of_input_features, no_of_dimensions, no_of_instances, prediction_dimension;

    Layer *ptr;

    prediction_dimension = output->getNoOfNeuron();
    no_of_dimensions = X.getNoOfDimensions();
    no_of_instances = X.getDimensions()[no_of_dimensions - 1];
    no_of_input_features = 0;

    for (i = 0; i < no_of_dimensions - 1; i++)
    {
        no_of_input_features += X.getDimensions()[i];
    }

    X_input = NDArray<double, 0>(no_of_dimensions - 1, X.getDimensions());
    y_predict = NDArray<double, 0>(2, prediction_dimension, X.getDimensions()[no_of_dimensions - 1]);

    for (i = 0; i < no_of_instances; i++)
    {
        X_input.initData(X.getData()[i * no_of_input_features]);

        ptr = input;
        ptr->initilizeInputData(X_input);
        search.search_param = "predict";
        ptr->searchDFS(search);

        ptr = output;
    }*/

    return y_predict;
}

Sequential::Sequential() : Model("sequential"){};

void Sequential::add(Layer *layer)
{
    if (output)
    {
        *(output) = layer;
        output = layer;
    }
    else
    {
        input = layer;
        output = layer;
    }
}

void Sequential::compile(std::string loss, std::string optimizer, std::string metrics)
{
    Layer *ptr = input;
    search.search_param = "compile";
    search.String = optimizer;
    ptr->searchDFS(search);

    switch (st_loss.loss[loss])
    {
    case this->st_loss.mean_squared_error:
        this->loss = new Mean_Squared_Error;
        break;
    case this->st_loss.categorical_crossentropy:
        this->loss = new Catagorical_Cross_Entropy;
        break;

    default:
        break;
    }

    switch (st_metric.metric[metrics])
    {
    case this->st_metric.accuracy:
        this->metric = new Accuracy;
        break;

    case this->st_metric.precision:
        break;

    case this->st_metric.recall:
        break;

    case this->st_metric.f1_score:
        break;

    default:
        break;
    }
}