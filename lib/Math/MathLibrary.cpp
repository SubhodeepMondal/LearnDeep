template <typename T, int typeFlag>
void NDMath<T, typeFlag>::recursive_sum(unsigned reduction_dim, unsigned index, unsigned *dimension_arr, NDMath<T, typeFlag> input, T *temp_input, NDMath<T, typeFlag> output)
{

    if (index < 3)
    {
        unsigned i, j, k, x_axis, y_axis, z_axis, stride, n_dim_size, input_index, output_index;
        T *input_ptr;
        T *output_ptr;

        // T *input = this->getData();

        x_axis = input.getDimensions()[0];
        y_axis = (input.getNoOfDimensions() > 1) ? input.getDimensions()[1] : 1;
        z_axis = (input.getNoOfDimensions() > 2) ? input.getDimensions()[2] : 1;

        input_ptr = input.getData();
        output_ptr = output.getData();

        input_index = output_index = 0;

        if (input.getNoOfDimensions() > 3)
        {
            n_dim_size = x_axis * y_axis * z_axis;
            for (i = 3; i < input.getNoOfDimensions(); i++)
            {
                input_index += n_dim_size * dimension_arr[i];
                n_dim_size *= input.getDimensions()[i];
            }

            n_dim_size = 1;
            for (i = 0; i < input.getNoOfDimensions(); i++)
            {
                if (i != reduction_dim)
                {
                    if (i < 3)
                        output_index *= n_dim_size;
                    else
                        output_index += n_dim_size * dimension_arr[i];

                    n_dim_size *= input.getDimensions()[i];
                }
            }
        }

        switch (reduction_dim)
        {
        case 0:
        {
            for (k = 0; k < x_axis; k++)
            {
                stride = 1;
                for (j = 0; j < z_axis; j++)
                    for (i = 0; i < y_axis; i++)
                        temp_input[i + j * y_axis] = input_ptr[i * x_axis + j * x_axis * y_axis + stride * k + input_index];
                cpu::__madd(output_ptr + output_index, temp_input, output_ptr + output_index, y_axis, z_axis);
            }
            break;
        }

        case 1:
        {

            for (k = 0; k < y_axis; k++)
            {
                stride = x_axis;
                for (j = 0; j < z_axis; j++)
                    for (i = 0; i < x_axis; i++)
                        temp_input[i + j * x_axis] = input_ptr[i + j * x_axis * y_axis + stride * k + input_index];
                cpu::__madd(output_ptr + output_index, temp_input, output_ptr + output_index, x_axis, z_axis);
            }

            break;
        }
        case 2:
        {
            for (k = 0; k < z_axis; k++)
            {
                T *temp_inp;
                stride = x_axis * y_axis;
                temp_inp = input_ptr + (stride * k + input_index);
                cpu::__madd(output_ptr + output_index, temp_inp, output_ptr + output_index, x_axis, y_axis);
            }
            break;
        }

        default:
        {
            for (k = 0; k < z_axis; k++)
            {
                T *temp_inp;
                stride = x_axis * y_axis;
                temp_inp = input_ptr + (stride * k + input_index);
                cpu::__madd(output_ptr + (output_index + stride * k), temp_inp, output_ptr + (output_index + stride * k), x_axis, y_axis);
            }
            break;
        }
        }
    }
    else
    {
        for (unsigned i = 0; i < input.getDimensions()[index]; i++)
        {
            dimension_arr[index] = i;
            recursive_sum(reduction_dim, index - 1, dimension_arr, input, temp_input, output);
        }
    }
}

template <typename T, int typeFlag>
NDMath<T, typeFlag> NDMath<T, typeFlag>::matrixMultiplication(const NDMath<double, 0> input)
{
    NDMath<T, typeFlag> output;
    unsigned i, j, no_of_dimensions, flag = 1;
    unsigned a_plan_dim, b_plan_dim, c_plan_dim;
    unsigned a_actual_index, b_actual_index, c_actual_index;
    unsigned dim_x, dim_y, dim_z;
    unsigned *output_dim;

    no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

    dim_x = input.getDimensions()[0];
    dim_y = NDMath<T, typeFlag>::getDimensions()[1];
    dim_z = NDMath<T, typeFlag>::getDimensions()[0];

    if (no_of_dimensions == input.getNoOfDimensions())
    {
        output_dim = new unsigned[no_of_dimensions];

        output_dim[0] = dim_x;
        output_dim[1] = dim_y;

        if (this->getDimensions()[0] == input.getDimensions()[1])
        {

            for (i = 2; i < no_of_dimensions; i++)
            {
                output_dim[i] = NDMath<T, typeFlag>::getDimensions()[i];
                if (NDMath<T, typeFlag>::getDimensions()[i] != input.getDimensions()[i])
                {
                    flag = 0;
                    break;
                }
            }
            if (flag)
            {

                output = NDMath<T, typeFlag>(no_of_dimensions, output_dim);
                unsigned *dimension_arr = new unsigned[this->getNoOfDimensions()];

                recursive_iterator(this->getNoOfDimensions() - 1, dimension_arr, input, output);

                delete[] dimension_arr;
                // output.printData();
            }
            else
            {
                std::cout << "Error!" << i << "th Dimension does not match with second matrix.\n";
                return NULL;
            }
        }
        else
        {
            std::cout << "Error! First matrix's row length does not match with second matrix column length.\n";
            return NULL;
        }
    }
    else
    {
        std::cout << "Dimension mismatch, First matrix doesn't have same no of dimension of second matrix.\n";
        return NULL;
    }

    return output;
}

template <typename T, int typeFlag>
NDMath<T, typeFlag> NDMath<T, typeFlag>::matrixMultiplication(const double scalerFactor)
{
    NDMath<T, typeFlag> output;

    unsigned no_of_dims, plan_size, dim_x, dim_y;

    no_of_dims = NDMath<T, typeFlag>::getNoOfDimensions();
    dim_x = NDMath<T, typeFlag>::getDimensions()[0];
    dim_y = NDMath<T, typeFlag>::getDimensions()[1];

    output = NDArray<T, typeFlag>(no_of_dims, this->getDimensions);
    if (no_of_dims < 3 && no_of_dims > 0)
    {
        cpu::__mscalermul(this->getData(), scalerFactor, output->getData(), dim_x, dim_y);
    }
    else
    {
        plan_size = 0;
        for (unsigned i = 2; i < no_of_dims; i++)
            for (unsigned j = 0; j < this->getDimensions()[i]; j++)
            {
                cpu::__mscalermul(this->getData() + plan_size, scalerFactor, this->getData(), dim_x, dim_y);
                plan_size += dim_x * dim_y;
            }
    }
}

template <typename T, int typeFlag>
NDMath<T, typeFlag> NDMath<T, typeFlag>::operator*(const NDMath<double, 0> input)
{
    NDMath<T, typeFlag> output;
    unsigned i, j, no_of_dimensions, flag = 1;
    unsigned a_plan_dim, b_plan_dim, c_plan_dim;
    unsigned a_actual_index, b_actual_index, c_actual_index;
    unsigned dim_x, dim_y, dim_z;
    unsigned *output_dim;

    no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

    dim_x = input.getDimensions()[0];
    dim_y = NDMath<T, typeFlag>::getDimensions()[1];
    dim_z = NDMath<T, typeFlag>::getDimensions()[0];

    if (no_of_dimensions == input.getNoOfDimensions())
    {
        output_dim = new unsigned[no_of_dimensions];

        output_dim[0] = dim_x;
        output_dim[1] = dim_y;

        std::cout << dim_x << " " << dim_y << "\n";

        for (i = 2; i < no_of_dimensions; i++)
        {
            output_dim[i] = NDMath<T, typeFlag>::getDimensions()[i];
            if (NDMath<T, typeFlag>::getDimensions()[i] != input.getDimensions()[i])
            {
                flag = 0;
                break;
            }
        }
        if (flag && this->getDimensions()[0] == input.getDimensions()[1])
        {

            output = NDMath<T, typeFlag>(no_of_dimensions, output_dim);

            if (no_of_dimensions < 3)
            {

                std::cout << "inside no_of_dim < 3\n";
                output.printDimensions();
                std::cout << "\n";
                std::cout << output.getDimensions()[0] << "\n";
                std::cout << output.getDimensions()[1] << "\n";
                cpu::__mmul(NDMath<T, typeFlag>::getData(), input.getData(), output.getData(), dim_x, dim_y, dim_z);
                // output.printData();
            }
            else
            {
                a_plan_dim = dim_y * dim_z;
                b_plan_dim = dim_x * dim_z;
                c_plan_dim = dim_x * dim_y;
                a_actual_index = b_actual_index = c_actual_index = 0;
                for (i = 2; i < no_of_dimensions; i++)
                {
                    for (j = 0; j < NDMath<T, typeFlag>::getDimensions()[i]; j++)
                    {
                        cpu::__mmul(NDMath<T, typeFlag>::getData() + a_actual_index, input.getData() + b_actual_index, output.getData() + c_actual_index, dim_x, dim_y, dim_z);
                        a_actual_index += a_plan_dim;
                        b_actual_index += b_plan_dim;
                        c_actual_index += c_plan_dim;
                    }
                }

                std::cout << "inside no_of_dim >= 3\n";
            }
        }
        output.printData();
    }

    output.printData();
    return output;
}

template <typename T, int typeFlag>
NDMath<T, typeFlag> NDMath<T, typeFlag>::matrixAddition(const NDMath<double, 0> input)
{
    NDMath<T, typeFlag> output;

    unsigned dim_x, dim_y, plane_offset, no_of_dimensions, flag;

    flag = 1;

    no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

    for (int i = 0; i < no_of_dimensions; i++)
        if (this->getDimensions()[i] != input.getDimensions()[i])
        {
            flag = 0;
            break;
        }
    if (flag)
    {
        dim_x = this->getDimensions()[0];
        dim_y = this->getDimensions()[1];
        plane_offset = 0;

        output = NDMath<T, typeFlag>(this->getNoOfDimensions(), this->getDimensions());

        if (no_of_dimensions < 3)
        {
            cpu::__madd(this->getData(), input.getData(), output.getData(), dim_x, dim_y);
        }
        else
        {
            for (int i = 2; i < no_of_dimensions; i++)
                for (int j = 0; j < this->getDimensions()[i]; j++)
                {
                    cpu::__madd(this->getData() + plane_offset, input.getData() + plane_offset, output.getData() + plane_offset, dim_x, dim_y);
                    plane_offset += dim_x * dim_y;
                }
        }
        return output;
    }
    else
    {
        std::cout << "Two metrix requires same shape to perform matrix addition, here matrix A ";
        NDMath<T, typeFlag>::printDimensions();
        std::cout << " and matrix B ";
        input.printDimensions();
        std::cout << " are of differenct shape.\n";
        return NULL;
    }
}

template <typename T, int typeFlag>
NDMath<T, typeFlag> NDMath<T, typeFlag>::matrixVectorAddition(const NDMath<double, 0> input)
{
    NDMath<T, typeFlag> output, temp_input;

    unsigned dim_x, dim_y, plane_offset, no_of_dimensions, flag;

    flag = 1;

    no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

    if (this->getDimensions()[0] != input.getDimensions()[0])
    {
        std::cout << "Two metrix requires same shape for x-axis to perform matrix addition, here matrix A ";
        NDMath<T, typeFlag>::printDimensions();
        std::cout << " and matrix B ";
        input.printDimensions();
        std::cout << " are of differenct shape on x-axis.\n";

        return NULL;
    }
    else
    {
        dim_x = this->getDimensions()[0];
        dim_y = this->getDimensions()[1];
        plane_offset = 0;

        output = NDMath<T, typeFlag>(this->getNoOfDimensions(), this->getDimensions());
        temp_input = NDMath<T, typeFlag>(dim_x, dim_y);

        for (unsigned i = 0; i < dim_y; i++)
            temp_input.initPartialData(i * dim_x, dim_x, input.getData());

        if (no_of_dimensions < 3)
        {
            cpu::__madd(this->getData(), temp_input.getData(), output.getData(), dim_x, dim_y);
        }
        else
        {
            for (int i = 2; i < no_of_dimensions; i++)
                for (int j = 0; j < this->getDimensions()[i]; j++)
                {
                    cpu::__madd(this->getData() + plane_offset, temp_input.getData() + plane_offset, output.getData() + plane_offset, dim_x, dim_y);
                    plane_offset += dim_x * dim_y;
                }
        }
        temp_input.destroy();
        return output;
    }
}

template <typename T, int typeFlag>
NDMath<T, typeFlag> NDMath<T, typeFlag>::operator+(const NDMath<double, 0> input)
{
    NDMath<T, typeFlag> output;

    unsigned dim_x, dim_y, plane_offset, no_of_dimensions, flag;

    flag = 1;

    no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

    for (int i = 0; i < no_of_dimensions; i++)
        if (this->getDimensions()[i] != input.getDimensions()[i])
        {
            flag = 0;
            break;
        }
    if (flag)
    {
        dim_x = this->getDimensions()[0];
        dim_y = this->getDimensions()[1];
        plane_offset = 0;

        output = NDMath<T, typeFlag>(this->getNoOfDimensions(), this->getDimensions());

        if (no_of_dimensions < 3)
        {
            cpu::__madd(this->getData(), input.getData(), output.getData(), dim_x, dim_y);
        }
        else
        {
            for (int i = 2; i < no_of_dimensions; i++)
                for (int j = 0; j < this->getDimensions()[i]; j++)
                {
                    cpu::__madd(this->getData() + plane_offset, input.getData() + plane_offset, output.getData() + plane_offset, dim_x, dim_y);
                    plane_offset += dim_x * dim_y;
                }
        }
        return output;
    }
    else
    {
        std::cout << "Two metrix requires same shape to perform matrix addition, here matrix A ";
        NDMath<T, typeFlag>::printDimensions();
        std::cout << " and matrix B ";
        input.printDimensions();
        std::cout << " are of differenct shape.\n";
        return output;
    }
}

template <typename T, int typeFlag>
NDMath<T, typeFlag> NDMath<T, typeFlag>::operator-(const NDMath<double, 0> input)
{
    NDMath<T, typeFlag> output;

    unsigned dim_x, dim_y, plane_offset, no_of_dimensions, flag;

    flag = 1;

    no_of_dimensions = NDMath<T, typeFlag>::getNoOfDimensions();

    for (int i = 0; i < no_of_dimensions; i++)
        if (this->getDimensions()[i] != input.getDimensions()[i])
        {
            flag = 0;
            break;
        }
    if (flag)
    {
        dim_x = this->getDimensions()[0];
        dim_y = this->getDimensions()[1];
        plane_offset = 0;

        output = NDMath<T, typeFlag>(this->getNoOfDimensions(), this->getDimensions());

        if (no_of_dimensions < 3)
        {
            cpu::__madd(this->getData(), input.getData(), output.getData(), dim_x, dim_y);
        }
        else
        {
            for (int i = 2; i < no_of_dimensions; i++)
                for (int j = 0; j < this->getDimensions()[i]; j++)
                {
                    cpu::__msub(this->getData() + plane_offset, input.getData() + plane_offset, output.getData() + plane_offset, dim_x, dim_y);
                    plane_offset += dim_x * dim_y;
                }
        }
        return output;
    }
    else
    {
        std::cout << "Two metrix requires same shape to perform matrix addition, here matrix A ";
        NDMath<T, typeFlag>::printDimensions();
        std::cout << " and matrix B ";
        input.printDimensions();
        std::cout << " are of differenct shape.\n";
        return NULL;
    }
}

template <typename T, int typeFlag>
void NDMath<T, typeFlag>::matrixTranspose()
{
    unsigned x, y;

    x = NDMath<T, typeFlag>::getDimensions()[0];
    y = NDMath<T, typeFlag>::getDimensions()[1];

    NDArray<T, typeFlag>::reshape(y, x);

    x = x + y;
    y = x - y;
    x = x - y;

    cpu::__mtranspose(this->getData(), this->getData(), x, y);
}

template <typename T, int typeFlag>
void NDMath<T, typeFlag>::reducesum(NDMath<T, typeFlag> *output)
{

    unsigned count = 0;
    unsigned *reduction_dims;
    unsigned *dims;
    NDMath<T, typeFlag> temp_output, temp_input;
    T *intermediate_input;

    arr_dims = new unsigned[this->getNoOfDimensions()];

    ptr = ptr_prev = head;

    while (ptr)
    {
        count++;
        ptr = ptr->next;
    }

    reduction_dims = new unsigned[count];
    ptr = head;

    for (unsigned i = 0; i < count; i++)
    {
        reduction_dims[i] = this->getNoOfDimensions() - ptr->value - 1;

        ptr = ptr->next;

        delete[] ptr_prev;

        ptr_prev = ptr;
    }

    // shorting dimensions using bubble short
    for (unsigned j = 0; j < count; j++)
        for (unsigned i = 0; i < count - j - 1; i++)
            if (reduction_dims[i] < reduction_dims[i + 1])
            {
                unsigned temp = reduction_dims[i];
                reduction_dims[i] = reduction_dims[i + 1];
                reduction_dims[i + 1] = temp;
            }

    // unsigned resulting_dim = this->getNoOfDimensions() - count;

    unsigned resulting_no_of_dims, flag, k;
    unsigned *resulting_dims;

    temp_output = NDMath<T, typeFlag>(this->getNoOfDimensions(), this->getDimensions());

    temp_input = NDMath<T, typeFlag>(this->getNoOfDimensions(), this->getDimensions());

    temp_input.initData(this->getData());

    intermediate_input = new T[this->getDimensions()[0] * this->getDimensions()[1] * this->getDimensions()[2]];

    resulting_dims = new unsigned[this->getNoOfDimensions()];

    for (unsigned i = 0; i < count; i++)
    {

        resulting_no_of_dims = temp_output.getNoOfDimensions() ? temp_output.getNoOfDimensions() - 1 : 1;

        if (temp_output.getNoOfDimensions() > 1)
        {
            k = 0;
            for (unsigned j = 0; j < temp_output.getNoOfDimensions(); j++)
                if (j != reduction_dims[i])
                    resulting_dims[k++] = temp_output.getDimensions()[j];
        }
        else
        {
            resulting_no_of_dims = 1;
            resulting_dims[0] = 1;
        }

        temp_output.destroy();
        temp_output = NDMath<T, typeFlag>(resulting_no_of_dims, resulting_dims);

        recursive_sum(reduction_dims[i], temp_input.getNoOfDimensions() - 1, arr_dims, temp_input, intermediate_input, temp_output);

        temp_input.destroy();
        temp_input = NDMath<T, typeFlag>(temp_output.getNoOfDimensions(), temp_output.getDimensions());

        temp_input.initData(temp_output.getData());
    }

    (*output) = NDMath<T, typeFlag>(temp_output.getNoOfDimensions(), temp_output.getDimensions());
    (*output).initData(temp_output.getData());
}

template <typename T, int typeFlag>
template <typename first_dim, typename... Args>
void NDMath<T, typeFlag>::reducesum(NDMath<T, typeFlag> *output, first_dim n, Args... args)
{
    if (n < this->getNoOfDimensions())
    {

        if (!head)
        {
            ptr = new struct arg_list;

            head = ptr_prev = ptr;

            head->value = n;
            head->next = NULL;
        }
        else
        {
            ptr = new struct arg_list;

            ptr->value = n;
            ptr->next = NULL;
            ptr_prev->next = ptr;
            ptr_prev = ptr;
        }
        reducesum(output, args...);
    }
    else
        std::cout << "Fatal error! reduction axis does not belong for the tensor\n";
}

template <typename T, int typeFlag>
template <typename... Args>
NDMath<T, typeFlag> NDMath<T, typeFlag>::sum(Args... args)
{
    head = NULL;
    NDMath<T, typeFlag> output;
    reducesum(&output, args...);
    return output;
}
