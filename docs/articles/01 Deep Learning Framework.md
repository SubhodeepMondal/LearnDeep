<p align="center">
  <img src="../images/Tensor Operations From Scratch.png"
       alt="LearnDeep — Deep Learning Framework From Scratch"
       width="100%">
</p>

<h1 align="center">LearnDeep</h1>

<p align="center">
  A Deep Learning Framework Built From Scratch in C++
</p>

#01 Deep Learning Framework from Scratch

Okay, with the motto and mission in mind, I've taken up this passion project of building my own deep learning framework from scratch, where I'll implement my own:

1. Tensors (strictly no `std::vector<>`).
2. Operations (`add()`, `matmul()`, `div()` — you get the idea).
3. Kernels for these operations (basic CPU, AVX2, and GPU kernels).
4. My own layers, optimizers, losses, and model framework.

## Why reinvent the wheel?

Why not? Just to learn.

If I jump straight into TensorFlow or PyTorch, a lot of the underlying implementation would go over my head. And I wouldn't get to learn how the kernels are actually written.

Before understanding *how*, I'd spend an enormous amount of time wondering *why*.

Simply put, that's the reason.

## Tensor 101

So, basically, a tensor is a multidimensional array. I mean, we already have scalar values, then vectors, then matrices, and then tensors generalize the idea to arbitrary dimensions. Einstein didn't invent tensors, but his work helped popularize them in physics.

### Why do we need tensors?

Just to do math, dumbass.

So, our prerequisite is a contiguous block of memory, along with another array that tells us the size of each dimension.

Ho la! We're good.

Like this:

```cpp
template<typename T>
class tensor {
    T* data;                 // Manual memory management.
    size_t no_of_dimensions;
    size_t* dimensions;

public:
    tensor();
    ~tensor();

    tensor add(tensor b);
    tensor matmul(tensor b);
};
```

No `std::unique_ptr`, no `std::vector`, because *"Hum toh mard hai, Jaskirat. We manage our own memory."*

`tf::tensor A(2, 3, 4);` is a 3D tensor with dimensions `(2, 3, 4)`. It contains 2 elements along the first dimension, 3 along the second, and 4 along the third.

Simple, easy peasy. No rocket science.

Yeah, you got me. My dimension ordering is different from the conventional matrix layout you'd expect in NumPy, TensorFlow, or PyTorch. In my framework, the leftmost dimension is the primary dimension, so a tensor with dimensions `(2, 3, 4)` would conventionally be represented as `(4, 3, 2)` in those libraries.

But I had my own thought process, and that's why I kept it this way. I can fix this with major refactoring.

He he.

Now, that's all good, but what can we do with tensors?

Maths, you dumbass. It's all math to begin with.

Like this:

```cpp
tf::tensor A, B, C;

A.tf_create(2, 3, 4);
B.tf_create(2, 3, 4);

A.tensor_of(data_1);
B.tensor_of(data_2);

C = A.add(B);
```

Or:

```cpp
A.tf_create(2, 3);
B.tf_create(4, 2);

C = A.matmul(B); // I know my axis orientation will boggle your mind,
                 // but we're stuck with it.

 // Result dimensions: (4, 3)
```

The matrix multiplication above follows my framework's custom dimension convention, so don't expect the usual NumPy-style shape notation.

### Now, how do you do the math?

BLAS, BLIS, LAPACK, cuBLAS, Eigen...

No, no, no. *"Kiuki, hum toh mard hai, Jaskirat."*

We're going to write our own kernels.

Instead of implementing every operation directly inside the tensor class, like this:

```cpp
class tensor {
    tensor add(tensor input_b) {
        tensor output;

        // Iterate over the elements and add them.

        return output;
    }
};
```

We can create an abstraction for operations where we keep the implementation and kernel dispatch logic together. This will help us further down the road.

1. Kernel dispatching, because we're targeting multiple kernels.
2. Kernel registration.
3. Automatic differentiation (autodiff).

The sleeping giant. He he.

Something along these lines:

```cpp
class ops {
public:
    virtual void compute() = 0;
    virtual void kernel_dispatcher() = 0;
};

class add : protected ops {
public:
    void compute() override;
    void kernel_dispatcher() override;
};
```

The code above is illustrative; the actual implementation will evolve as the framework grows.

You get the idea.

### Now that we have the basic framework ready, what's left to do?

Testing.

NumPy is our guide. I'll generate test data, calculate the expected results using NumPy, and compare them against the output of my own implementation.

And here we go. Here are the results.

*[Insert screenshots of the test suite and validation results here.]*

To be continued...
