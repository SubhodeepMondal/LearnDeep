#pragma ONCE

template <typename T, int typeFlag>
class node
{
    std::string operation;
    NDMath<T, typeFlag> operand_a, operand_b;
    NDMath<T, typeFlag> output;

    node **input_nodes, **output_nodes;
};
