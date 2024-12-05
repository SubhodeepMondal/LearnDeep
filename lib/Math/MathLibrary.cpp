template <typename T, int typeFlag>
void NDMath<T, typeFlag>::recursive_iterator(unsigned index,
                                             unsigned *dimension_arr,
                                             NDMath<T, typeFlag> input,
                                             NDMath<T, typeFlag> &output,
                                             void (*__kernel)(double **, unsigned *),
                                             std::string function_name,
                                             unsigned *ui_arr,
                                             double *dl_arr,
                                             NDMath<T, typeFlag> misc_arr)
{
    if (index < 2)
    {
        unsigned i, inpA_x, inpA_y, inpB_x, inpB_y, out_x, out_y;
        unsigned a_plane_size, b_plane_size, c_plane_size, a_index, b_index, c_index;

        inpA_x = (this->getNoOfDimensions() > 0) ? NDMath<T, typeFlag>::getDimensions()[0] : 1;
        inpA_y = (this->getNoOfDimensions() > 1) ? NDMath<T, typeFlag>::getDimensions()[1] : 1;

        inpB_x = (input.getNoOfDimensions() > 0) ? input.getDimensions()[0] : 1;
        inpB_y = (input.getNoOfDimensions() > 1) ? input.getDimensions()[1] : 1;

        out_x = (output.getNoOfDimensions() > 0) ? output.getDimensions()[0] : 1;
        out_y = (output.getNoOfDimensions() > 1) ? output.getDimensions()[1] : 1;

        a_plane_size = inpA_x * inpA_y;
        b_plane_size = inpB_x * inpB_y;
        c_plane_size = out_x * out_y;

        a_index = b_index = c_index = 0;
        // std::cout << "a index: " << a_plane_size << " b index: " << b_plane_size << " c index: " << c_plane_size << "\n";
        if (input.getNoOfDimensions() > 2)
            for (i = 2; i < input.getNoOfDimensions(); i++)
            {
                std::cout << dimension_arr[i] << " ";
                a_index += a_plane_size * dimension_arr[i];
                b_index += b_plane_size * dimension_arr[i];
                c_index += c_plane_size * dimension_arr[i];

                a_plane_size *= this->getDimensions()[i];
                b_plane_size *= input.getDimensions()[i];
                c_plane_size *= output.getDimensions()[i];
                // std::cout << "a index: " << a_index << " b index: " << b_index << " c index: " << c_index << "\n";
            }

        // std::cout << "a index: " << a_index << " b index: " << b_index << " c index: " << c_index << "\n";
        // std::cout << input.getData() << "  " << output.getData() << "\n";
        switch (fx_name.function_name[function_name])
        {
        case this->fx_name.matrix_multiplication:
        {
            /* code */
            unsigned a[3];
            double *ptr[3];

            a[0] = inpB_x;
            a[1] = inpB_y;
            a[2] = inpA_y;

            ptr[0] = NDMath<T, typeFlag>::getData() + a_index;
            ptr[1] = input.getData() + b_index;
            ptr[2] = output.getData() + c_index;
            __kernel(ptr, a);

            break;
        }
        case this->fx_name.matrix_addition:
        {
            /* code */
            unsigned a[2];
            double *ptr[3];

            a[0] = inpA_x;
            a[1] = inpA_y;

            ptr[0] = NDMath<T, typeFlag>::getData() + a_index;
            ptr[1] = input.getData() + b_index;
            ptr[2] = output.getData() + c_index;
            __kernel(ptr, a);

            break;
        }
        case this->fx_name.matrix_power:
        {
            int j;
            // int exponent = ui_arr[0];
            unsigned a[2];
            double *ptr[3];


            a[0] = inpA_x;
            a[1] = inpA_y;

            ptr[0] = this->getData() + a_index;
            ptr[1] = input.getData() + b_index;
            ptr[2] = output.getData() + c_index;


            cpu::__melementwisemul(ptr, a);

            break;
        }

        default:
            break;
        }
    }
    else
    {
        // std::cout << "inside else\n";
        for (unsigned i = 0; i < NDMath<T, typeFlag>::getDimensions()[index]; i++)
        {
            dimension_arr[index] = i;
            recursive_iterator(index - 1, dimension_arr, input, output, __kernel, function_name, NULL, NULL, NULL);
        }
    }
}

template <typename T, int typeFlag>
void NDMath<T, typeFlag>::recursive_sum(unsigned index,
                                        unsigned *dimension_arr,
                                        NDMath<T, typeFlag> input,
                                        NDMath<T, typeFlag> &output,
                                        unsigned reduction_dim,
                                        T *temp_input)
{

    if (index < 3)
    {
        unsigned i, j, k;
        unsigned x_axis, y_axis, z_axis, stride, n_dim_size;
        unsigned input_index, output_index;
        T *input_ptr, *output_ptr, *temp_inp;
        double *ptr[3];
        unsigned a[2];

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

            ptr[0] = ptr[2] = output_ptr + output_index;
            a[0] = x_axis;
            a[1] = z_axis;

            for (k = 0; k < x_axis; k++)
            {
                stride = 1;
                for (j = 0; j < z_axis; j++)
                    for (i = 0; i < y_axis; i++)
                        temp_input[i + j * y_axis] = input_ptr[i * x_axis + j * x_axis * y_axis + stride * k + input_index];

                ptr[1] = temp_input;
                cpu::__madd(ptr, a);
                // cpu::__madd(output_ptr + output_index, temp_input, output_ptr + output_index, y_axis, z_axis);
            }
            break;
        }

        case 1:
        {

            ptr[0] = ptr[2] = output_ptr + output_index;
            a[0] = x_axis;
            a[1] = z_axis;
            for (k = 0; k < y_axis; k++)
            {
                stride = x_axis;
                for (j = 0; j < z_axis; j++)
                    for (i = 0; i < x_axis; i++)
                        temp_input[i + j * x_axis] = input_ptr[i + j * x_axis * y_axis + stride * k + input_index];

                ptr[1] = temp_input;
                cpu::__madd(ptr, a);
                // cpu::__madd(output_ptr + output_index, temp_input, output_ptr + output_index, x_axis, z_axis);
            }

            break;
        }
        case 2:
        {

            ptr[0] = ptr[2] = output_ptr + output_index;
            a[0] = x_axis;
            a[1] = y_axis;

            for (k = 0; k < z_axis; k++)
            {
                stride = x_axis * y_axis;
                temp_input = input_ptr + (stride * k + input_index);
                ptr[1] = temp_input;

                cpu::__madd(ptr, a);

                // for (int j = 0; j < y_axis; j++)
                //     for (int i = 0; i < x_axis; i++)
                //         std::cout << output_ptr[i + j * x_axis] << " ";

                // cpu::__madd(output_ptr + output_index, temp_inp, output_ptr + output_index, x_axis, y_axis);
            }
            break;
        }

        default:
        {
            a[0] = x_axis;
            a[1] = y_axis;
            for (k = 0; k < z_axis; k++)
            {
                stride = x_axis * y_axis;

                ptr[0] = ptr[2] = output_ptr + (output_index + stride * k);

                temp_inp = input_ptr + (stride * k + input_index);
                ptr[1] = temp_inp;

                cpu::__madd(ptr, a);
                // cpu::__madd(output_ptr + (output_index + stride * k), temp_inp, output_ptr + (output_index + stride * k), x_axis, y_axis);
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
            recursive_sum(index - 1, dimension_arr, input, output, reduction_dim, temp_input);
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

                recursive_iterator(this->getNoOfDimensions() - 1, dimension_arr, input, output, cpu::__mmul, "matrix_multiplication", NULL, NULL, NULL);

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

    output = NDMath<T, typeFlag>(no_of_dims, this->getDimensions);
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
        unsigned *arr = new unsigned[this->getNoOfDimensions()];

        recursive_iterator(this->getNoOfDimensions() - 1, arr, input, output, cpu::__madd, "matrix_addition", NULL, NULL, NULL);
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
            // cpu::__madd(this->getData(), input.getData(), output.getData(), dim_x, dim_y);
        }
        else
        {
            for (int i = 2; i < no_of_dimensions; i++)
                for (int j = 0; j < this->getDimensions()[i]; j++)
                {
                    // cpu::__madd(this->getData() + plane_offset, input.getData() + plane_offset, output.getData() + plane_offset, dim_x, dim_y);
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

        unsigned *dimension_arr = new unsigned[this->getNoOfDimensions()];

        recursive_iterator(this->getNoOfDimensions() - 1, dimension_arr, input, output, cpu::__msub, "matrix_multiplication", NULL, NULL, NULL);

        // if (no_of_dimensions < 3)
        // {
        //     cpu::__madd(this->getData(), input.getData(), output.getData(), dim_x, dim_y);
        // }
        // else
        // {
        //     for (int i = 2; i < no_of_dimensions; i++)
        //         for (int j = 0; j < this->getDimensions()[i]; j++)
        //         {
        //             cpu::__msub(this->getData() + plane_offset, input.getData() + plane_offset, output.getData() + plane_offset, dim_x, dim_y);
        //             plane_offset += dim_x * dim_y;
        //         }
        // }

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
NDMath<T, typeFlag> NDMath<T, typeFlag>::matrixpow(const unsigned exponent)
{
    NDMath<T, typeFlag> output;
    return output;
}

template <typename T, int typeFlag>
void NDMath<T, typeFlag>::matrixTranspose()
{
    unsigned x, y;

    x = NDMath<T, typeFlag>::getDimensions()[0];
    y = NDMath<T, typeFlag>::getDimensions()[1];

    NDMath<T, typeFlag>::reshape(y, x);

    x = x + y;
    y = x - y;
    x = x - y;

    cpu::__mtranspose(this->getData(), this->getData(), x, y);
}

template <typename T, int typeFlag>
void NDMath<T, typeFlag>::reducesum(NDMath<T, typeFlag> &output)
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

    temp_output.setObjName("temp_output");
    temp_input.setObjName("temp_input");

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

        // temp_output.destroy();
        temp_output = NDMath<T, typeFlag>(resulting_no_of_dims, resulting_dims);
        temp_output.initData(0.0);

        std::cout << "Reducing dimension:" << reduction_dims[i] << "\n";

        recursive_sum(temp_input.getNoOfDimensions() - 1, arr_dims, temp_input, temp_output, reduction_dims[i], intermediate_input);

        temp_input = temp_output;
    }

    output = temp_output;

    /*

    This destroy is not necessary, but it is here to fix a bug.
    when we're setting A tensor of dim 3 and each dimension as (4, 3, 4) (last dimension >3)
    and doing a reduction sum on all dimensions its destroyer throwing an error!
    Error: free(): invalid next size (fast):
    */
    temp_output.destroy();

    delete[] resulting_dims;
    delete[] intermediate_input;
}

template <typename T, int typeFlag>
template <typename first_dim, typename... Args>
void NDMath<T, typeFlag>::reducesum(NDMath<T, typeFlag> &output, first_dim n, Args... args)
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
NDMath<T, typeFlag> NDMath<T, typeFlag>::reducesum(Args... args)
{
    head = NULL;
    NDMath<T, typeFlag> output;
    reducesum(output, args...);
    return output;
}

template <typename T, int typeFlag>
NDMath<T, typeFlag> NDMath<T, typeFlag>::power(const unsigned exponent)
{
    unsigned i, *arr;

    NDMath<T, typeFlag> output(this->getNoOfDimensions(), this->getDimensions());

    if (exponent == 0)
        output.initData(1);
    else if (exponent > 0)
    {
        output.initData(this->getData());
        arr = new unsigned[this->getNoOfDimensions()];

        // std::cout << output.getData() << "\n";
        for (i = 1; i < exponent; i++)
            recursive_iterator(this->getNoOfDimensions() - 1, arr, output, output, cpu::__melementwisemul, "matrix_power", NULL, NULL, NULL);
        delete[] arr;
    }

    return output;
}