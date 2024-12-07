#pragma ONCE

template <typename T, int typeFlag>
class graph
{

    typedef struct node
    {
        std::string operation;
        NDMath<T, typeFlag> &operand_a, &operand_b;
        NDMath<T, typeFlag> &output;

        node *input_nodes, *back_prop_nodes, *next_node;
        unsigned input_node_count;
    } node;

    node *head, *current_node;
    graph() : head(NULL), current_node(NULL) {}

    void addcomputenode(NDMath<T, typeFlag> &oper_a, NDMath<T, typeFlag> &oper_b, std::string ops)
    {
        node *ptr = new node;

        node->operand_a = oper_a;
        node->operand_b = oper_b;
        node->operation = ops;
        next_node = input_nodes = back_prop_nodes = NULL;

        if (head == NULL)
            head = current_node = ptr;
        else
        {
            current_node->next_node = ptr;
            ptr->input_nodes = current_node;
            current_node = ptr;
        }
    }
};