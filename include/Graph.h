#pragma ONCE

template <typename T, int typeFlag>
class NDMath;

template <typename T, int typeFlag>
class Graph
{

    typedef struct node
    {
        Ops<T, typeFlag> *ops;

        node *next_node;
        node **output_nodes;
        node **input_nodes;
        unsigned input_node_count, output_node_count;
        unsigned input_node_index, output_node_index;

        bool ismemoryallocated;

        node()
        {
            next_node = NULL;
            output_nodes = input_nodes = NULL;
            input_node_count = input_node_index = output_node_count = output_node_index = 0;
            ismemoryallocated = false;
        }

        void addinputnode(node *input_ptr)
        {
            input_nodes[input_node_index++] = input_ptr;
        }

        void addoutputnode(node *output_ptr)
        {
            output_nodes[output_node_index++] = output_ptr;
        }

        void incinputnodecount() { input_node_count++; }
        void incoutputnodecount() { output_node_count++; }

        void allocatememoryforinputouputnode()
        {
            std::cout << "This node has " << input_node_count << " & " << output_node_count << " input and output nodes.\n";
            output_nodes = new node *[output_node_count];
            input_nodes = new node *[input_node_count];
        }

    } node;

    node *head, *current_node;

public:
    Graph() : head(NULL), current_node(NULL) {}
    void addcomputenode(NDMath<T, typeFlag> *oper_a, NDMath<T, typeFlag> *oper_b, Ops<T, typeFlag> *ops)
    {
        node *ptr = new node;

        // ptr->operand_a = oper_a;
        // ptr->operand_b = oper_b;
        ptr->ops = ops;
        ptr->input_node_count = ops->getnoofinputs();

        if (head == NULL)
        {
            head = current_node = ptr;
        }
        else
        {
            current_node->next_node = ptr;
            current_node = ptr;
        }
    }

    void optimize()
    {
        node *ptr = head->next_node;
        node *prev_ptr = head;
        unsigned i;

        // std::cout << ptr << "\n";
        // std::cout << prev_ptr << "\n";

        // Pass 1: to get input and output count for each node
        while (ptr)
        {
            for (i = 0; i < ptr->input_node_count; i++)
            {
                // std::cout << ptr->ops->getinputs()[i] << "\n";
                prev_ptr = head;
                while (prev_ptr != ptr)
                {

                    // std::cout << prev_ptr->ops->getoutput() << "\n";
                    if (ptr->ops->getinputs()[i] == prev_ptr->ops->getoutput())
                    {
                        // ptr->incinputnodecount();
                        prev_ptr->incoutputnodecount();
                    }
                    prev_ptr = prev_ptr->next_node;
                }
                // std::cout << "\n";
            }
            ptr = ptr->next_node;
        }

        // pass 2: to allocate memory for each node
        ptr = head;
        while (ptr)
        {
            ptr->allocatememoryforinputouputnode();
            ptr = ptr->next_node;
        }

        // pass 3: to connect all the incoming and outgoing nodes
        ptr = head;
        while (ptr)
        {
            for (i = 0; i < ptr->input_node_count; i++)
            {
                prev_ptr = head;
                while (prev_ptr != ptr)
                {
                    if (ptr->ops->getinputs()[i] == prev_ptr->ops->getoutput())
                    {
                        ptr->addinputnode(prev_ptr);
                        prev_ptr->addoutputnode(ptr);
                    }
                    prev_ptr = prev_ptr->next_node;
                }
            }
            ptr = ptr->next_node;
        }
    }

    void outputnode(NDMath<T, typeFlag> *output)
    {
        // current_node->output = output;
         return current_node->ops->initilizeoutput(output);
    }
    static void printnode(node *ptr)
    {
        ptr->ops->printinputs();
        ptr->ops->printoutput();
    }

    static void executionwrapper(node *ptr)
    {
        ptr->ops->compute();
    }

    void traversenode()
    {
        dfs(head, printnode);
    }

    void dfs(node *ptr, void (*func)(node *))
    {
        func(ptr);

        for (unsigned i = 0; i < ptr->output_node_count; i++)
        {
            dfs(ptr->output_nodes[i], func);
        }
    }

    void execute()
    {
        dfs(head, executionwrapper);
    }
};