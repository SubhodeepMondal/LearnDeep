#include <map>
#include "NDynamicArray.h"
#include "MathLibrary.h"
#include "Activations.h"
#include "OptimizationType.h"
#include "Optimizers.h"
#include "Forward_Propagation.h"
#include "Layers.h"


void Layer::operator=(Layer *layer)
{
    Layer_ptr *out_lyr_ptr, *out_prev_lyr_ptr;
    Layer_ptr *in_lyr_ptr, *in_prev_lyr_ptr;
    Layer_ptr *in_ptr, *out_ptr;

    in_ptr = new Layer_ptr;
    out_ptr = new Layer_ptr;

    // For outgoing edge
    if (out_vertices) // if out vertex is not NULL
    {

        out_prev_lyr_ptr = out_lyr_ptr = out_vertices;

        while (out_lyr_ptr)
        {
            if (out_lyr_ptr != out_prev_lyr_ptr)
                out_prev_lyr_ptr = out_prev_lyr_ptr->next;
            out_lyr_ptr = out_lyr_ptr->next;
        }

        out_ptr->layer = layer;
        out_ptr->previous = out_prev_lyr_ptr;
        out_prev_lyr_ptr->next = out_ptr;

        // std::cout << "out vertex tail: " << out_ptr << "\n";
    }
    else // if out vertex is NULL
    {
        out_ptr->layer = layer;
        out_vertices = out_ptr;
        // std::cout << "out vertex head: " << out_ptr << "\n";
    }

    if (layer->in_vertices)
    {
        in_prev_lyr_ptr = in_lyr_ptr = layer->in_vertices;

        while (in_lyr_ptr)
        {
            if (in_lyr_ptr != in_prev_lyr_ptr)
                in_prev_lyr_ptr = in_prev_lyr_ptr->next;
            in_lyr_ptr = in_lyr_ptr->next;
        }

        in_ptr->layer = this;
        in_ptr->previous = in_prev_lyr_ptr;
        in_prev_lyr_ptr->next = in_ptr;
        // std::cout << "in vertex tail:" << in_ptr << "\n";
    }
    else
    {
        in_ptr->layer = this;
        layer->in_vertices = in_ptr;
        // std::cout << "in vertex head:" << in_ptr << "\n";
    }
}

NDArray<double, 1> Layer::getOutputDataGPU()
{
    NDArray<double, 1> output;
    return output;
}

return_parameter Layer::searchDFS(search_parameter search)
{
    return_parameter return_param;
    return return_param;
}

void Layer::searchBFS(search_parameter search)
{
    
}