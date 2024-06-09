#include <iostream>
#include<map>
#include "DeepLearning.h"

void generatorFunctionLinear(NDArray<double, 0> x, NDArray<double, 0> y)
{
    unsigned no_of_sample;
    double *ptr1, *ptr2;


    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    no_of_sample = x.getDimensions()[1];

    ptr1 = x.getData();
    ptr2 = y.getData();

    for(int i = 0; i< no_of_sample ; i++)
    {
        ptr1[i] = distribution(generator);

        if(ptr1[i]< 0.5)
            ptr2[i] = 0.5 * ptr1[i] + 1.5;
        else
            ptr2[i] = -0.3 * ptr1[i] + 1.9;

    }


}

void generatorFunction1(NDArray<double, 0> x, NDArray<double, 0> y, unsigned no_of_feature, unsigned no_of_sample)
{
    unsigned i, j, lin_index;
    float angle;
    double *ptrA = x.getData();
    double *ptrB = y.getData();

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0f, 1.0f);
    for (i = 0; i < no_of_sample; i++)
    {
        for (j = 0; j < no_of_feature; j++)
        {
            lin_index = j + i * no_of_feature;
            angle = distribution(generator);
            // ptrA[i] = angle * (3.14159 / 180.0f);
            // ptrB[i] = sin(ptrA[i]);
            ptrA[lin_index] = angle;
            switch (j)
            {
            case 0:
                ptrB[i] = 1.5 * angle + 0.25;
                break;
            case 1:
                ptrB[i] += 0.75 * angle;
                break;
            case 2:
                ptrB[i] += 0.25 * angle;
                break;
            case 3:
                ptrB[i] += 1.75 * angle;
                break;
            case 4:
                ptrB[i] += 2.35 * angle;
                break;
            case 5:
                ptrB[i] += 3.05 * angle;
                break;
            case 6:
                ptrB[i] += 1.12 * angle;
                break;
            case 7:
                ptrB[i] += 4.012 * angle;
                break;

            default:
                break;
            }
        }
    }
}

void generatorFunctionSinWave(NDArray<double, 0> x, NDArray<double, 0> y)
{
    unsigned i;
    double angle;
    unsigned no_of_samples;
    double *ptrA, *ptrB;

    no_of_samples = x.getDimensions()[1];
    ptrA = x.getData();
    ptrB = y.getData();

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0f, 180.0f);

    for (i = 0; i < no_of_samples; i++)
    {
        angle = distribution(generator);
        ptrA[i] = angle * (3.14159 / 180.0f);
        ptrB[i] = sin(ptrA[i]) * sin(ptrA[i] - 0.45) * sin(ptrA[i] + 1.25);
    }
}

void generatorFunctionCluster(NDArray<double, 0> x, NDArray<double, 0> y)
{
    double x1, y1;
    unsigned i, j, no_of_features, no_of_samples;
    double *ptrA, *ptrB;

    no_of_features = x.getDimensions()[0];
    no_of_samples = x.getDimensions()[1];

    ptrA = x.getData();
    ptrB = y.getData();

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 5.0);

    i = 0;

    std::cout << "Generating cluster of data: .............\n";

    while (i < no_of_samples)
    {
        x1 = distribution(generator);
        y1 = distribution(generator);

        if (y1 < (1.5 * x1 - 3) && y1 > (-0.5 * x1 + 2))
        {
            j = 0;
            ptrA[i * no_of_features + j] =x1;
            j += 1;
            ptrA[i * no_of_features + j] = y1;

            j = 0;
            ptrB[i * no_of_features + j] = 1;
            j+= 1;
            ptrB[i * no_of_features + j] = 0;
            i++;
        }

        
        else if (y1 > (1.5 * x1 - 3) && y1 < (-0.5 * x1 + 2))
        {
            j = 0;
            ptrA[i * no_of_features + j] = x1;
            j+= 1;
            ptrA[i * no_of_features + j] = y1;

            j = 0;
            ptrB[i * no_of_features + j] = 0;
            j+= 1;
            ptrB[i * no_of_features + j] = 1;
            i++;
        }
    }
}

int main()
{
    NDMath math;
    freopen("io/input.txt", "r", stdin);
    freopen("io/DLearning.csv", "w", stdout);
    Metric *metric = new Accuracy;
    unsigned no_of_feature, no_of_sample, epochs, batch_size;
    NDArray<double, 1> y_predict_proba, y_predict, accuracy;
    no_of_sample = 1000;
    no_of_feature = 1;

    std::cin >> epochs;
    std::cin >> batch_size;

    NDArray<double, 0> X_train(2, no_of_feature, no_of_sample);
    NDArray<double, 0> y_train(2, 1, no_of_sample);


    NDArray<double, 0> X_test(2, no_of_feature, 100);
    NDArray<double, 0> y_test(2, 1, 100);



    // X_train.printData();
    // y_train.printData();

    generatorFunctionLinear(X_train,y_train);
    generatorFunctionLinear(X_test, y_test);
    // generatorFunctionCluster(X_train, y_train);
    // generatorFunctionCluster(X_test, y_test);

    // X_test.printData();
    // y_test.printData();

    NDArray<double, 0> input_shape(1, &no_of_feature);

    Model *model = new Sequential();
    model->add(new Dense{1, input_shape, "relu", "Dense_1"});
    model->add(new Dense{2, "relu", "Dense_2"});
    model->add(new Dense{1, "linear", "Dense_3"});
    model->compile("mean_squared_error", "adagrad", "accuracy");
    model->summary();

    model->fit(X_train, y_train, epochs, batch_size);

    // y_predict_proba = NDArray<double, 1>(y_test.getNoOfDimensions(), y_test.getDimensions());
    // y_predict = NDArray<double, 1>(y_test.getNoOfDimensions(), y_test.getDimensions());
    // accuracy = NDArray<double, 1>(1,1);
    // y_predict_proba = model->predict(X_test);
    // math.argMax(y_predict_proba, y_predict, NULL);
    // metric->accuracy(y_predict, y_test, accuracy, NULL);
    // accuracy.printData();
}