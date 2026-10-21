# Ececution Modes

[GitHub Repository Link]

## Graph Mode Execution

What are eager and graph mode execution?

Let's say we have the following chain of operations:

```cpp
c = a + b;
e = c * d;
f = e * a / d;
```

In eager mode, operations are executed as soon as we encounter them. We have `a` and `b`, so we execute `a + b`, store the result in `c`, and proceed to the next operation. Simple, effective, and easy to understand.

In fact, this is how we generally interact with programming languages, calculators, and many other computational systems.

However, this simplicity can become a limitation when we start building a deep learning framework.

Consider the following questions:

1. What if our inputs are tensors, and their actual values are not available when we define the operations?
2. What if we need to execute the same sequence of operations repeatedly?
3. What if we want to optimize the execution before running the operations?
4. What if we need to calculate partial derivatives of the operands and parameters for backpropagation?

This is where graph mode execution becomes useful.

Instead of immediately executing every operation we encounter, we construct a DAG (Directed Acyclic Graph). The graph represents the computational dependencies between operands and operations. Once the graph has been constructed and the required data is available, we trigger its execution using something like `run()` or `compute()`.

Let's explore why this approach is useful.

### Why Graph Mode?

**1. Separating graph construction from execution**

One of the biggest advantages of graph mode is that we can separate the definition of a computation from its actual execution.

We can construct the computational graph first, populate the tensors with data later, and then execute the graph.

This separation is particularly useful when the same computation needs to be executed repeatedly with different input values. Depending on the framework's implementation, we can reuse the graph structure instead of rebuilding the entire sequence of operations every time.

It also creates opportunities for more efficient memory management, such as reusing allocated buffers across repeated executions.

**2. Opportunities for graph optimization**

Once the computational structure is available, we can inspect it before execution and identify opportunities for optimization.

For example, consider the expression:

```cpp
f = (a * b) / b;
```

Under appropriate conditions, this expression can be simplified to:

```cpp
f = a;
```

A graph optimizer can identify such opportunities, eliminate redundant operations, and potentially reduce the amount of computation required.

More advanced optimizations can include operation fusion, eliminating computations whose results are never used, and planning memory reuse.

Of course, these optimizations must preserve the mathematical semantics of the original computation. Graph mode provides the structure needed to perform them; the framework still needs to implement the optimization algorithms.

**3. Automatic differentiation and backpropagation**

This is one of the most important reasons I wanted a computational graph in my framework.

During model training, we need to calculate gradients of the loss with respect to trainable parameters so that the optimizer can update them.

Since the graph preserves the dependencies between operations, we can traverse it in reverse order and apply the corresponding derivative rules to calculate gradients.

Consider an operand that contributes to multiple downstream operations. Its gradient must account for contributions from all those paths. This requires accumulating the gradients as we propagate backward through the graph.

A graph representation makes these dependencies explicit and provides a natural foundation for implementing reverse-mode automatic differentiation.

Graph mode is not the only way to implement autodiff, but it is the approach I chose for my framework.

### How Did I Implement It?

Now that we understand the motivation, a few implementation questions remain:

1. How does the framework determine whether an operation should execute immediately or be recorded in a graph?
2. Who manages the lifetime of tensors and the graph during execution?
3. How do we trigger graph execution?

For the graph execution mechanism, I took inspiration from TensorFlow 1.x and its graph-and-session model. My implementation follows a similar idea: create a graph context, record operations within that context, and explicitly trigger execution.

The implementation is adapted to the design choices and constraints of my C++ framework.

Let's look at the main components.

### 1. GraphContext

The first component is `GraphContext`.

Its responsibility is to establish an active graph context and provide an entry point for executing the graph. Operations performed within this context can be recorded in the active graph instead of being executed immediately.

A simplified representation looks like this:

```cpp
class GraphContext {
    Graph* graph;

public:
    GraphContext() {
        graph = new Graph();
        GraphManager::instance().push(graph);
    }

    void run() {
        graph->compute();
    }

    // The actual implementation must also handle
    // graph cleanup and context deactivation.
};
```

This is a simplified illustration of the design, not the complete implementation.

The important idea is that the context manages the period during which operations are recorded. Its lifecycle also needs to be handled correctly so that the active graph is removed from the manager and resources are released when the context ends.

This scoped approach gives us a foundation for extending graph management in the future.

### 2. GraphManager

`GraphManager` is a singleton responsible for managing active graph contexts.

When a graph context is created, it registers its graph with the manager. Operations can then identify the active graph and register themselves accordingly.

A stack-based design can also support nested contexts, provided that the implementation explicitly defines how nested graphs interact and how contexts are restored when an inner scope ends.

[GraphManager Source Code Link]

### 3. Graph

The `Graph` class contains the actual data structure representing the computational graph.

It is responsible for the core graph operations, including:

1. Node registration.
2. Dependency and edge creation.
3. Graph optimization.
4. Graph execution.

By keeping these responsibilities inside the graph abstraction, we can extend the execution engine without unnecessarily coupling graph management to the tensor API.

[Graph Source Code Link]

### Putting Everything Together

Now let's see how the two execution modes look when using my framework.

### Eager Mode Execution

```cpp
tf::tensor A, B, C;

A.tf_create(4, 3, 2);
B.tf_create(4, 3, 2);

A.tensor_of(data_1);
B.tensor_of(data_2);

// Eager mode: execute the operation immediately
C = A.add(B);
```

In eager mode, the operands are populated with data before the addition is executed. The result is available as soon as the operation completes.

### Graph Mode Execution

```cpp
tf::tensor A, B, C;

A.tf_create(4, 3, 2);
B.tf_create(4, 3, 2);

// Graph mode execution
{
    tf::GraphContext ctx;

    C = A.add(B);  // Record the operation in the DAG

    A.tensor_of(data_1);
    B.tensor_of(data_2);

    ctx.run();     // Execute the recorded graph
}
```

Here, the addition is recorded before the actual input values are populated. The graph is executed only when we explicitly call `ctx.run()`.

Notice the difference: we have introduced a few additional lines of code to obtain the same result.

So, why bother?

Because this separation between graph construction and execution gives us a foundation for features such as graph reuse, execution planning, and reverse-mode automatic differentiation.

And that is where the real benefits begin to emerge as we move from individual tensor operations to complete model training.

[Add Supporting Screenshots: GraphContext, GraphManager, DAG Representation, and Unit Test Results]

**To be continued...**