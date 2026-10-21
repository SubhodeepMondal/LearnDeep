#02 Deep Learning Framework from Scratch

# Operations

Are tensor operations simply mathematical operations? Think again.

The answer is yes, but they're doing more than just math. Sometimes they're shuffling data, sometimes masking, sometimes transposing. They're handling different kinds of operations: logical operations, data relocation, and, of course, mathematical operations.

## Basic Math Ops: `add()`, `sub()`, `mul()`...

These math ops are pretty straightforward because, most of the time, the elements are accessed contiguously. On GPUs, this can also mean coalesced memory access.

The implementation is simple:

```cpp
for (each element i) {
    output[i] = ops(input_a[i], input_b[i]);
}
```

I think we can do better by unrolling the loop:

```cpp
for (each row) {
    for (each group of 8 elements) {
        output[i]     = input_a[i]     + input_b[i];
        output[i + 1] = input_a[i + 1] + input_b[i + 1];
        output[i + 2] = input_a[i + 2] + input_b[i + 2];
        output[i + 3] = input_a[i + 3] + input_b[i + 3];
        output[i + 4] = input_a[i + 4] + input_b[i + 4];
        output[i + 5] = input_a[i + 5] + input_b[i + 5];
        output[i + 6] = input_a[i + 6] + input_b[i + 6];
        output[i + 7] = input_a[i + 7] + input_b[i + 7];
    }

    // Handle the remaining elements.
}
```

Once we do this, the idea can be transferred to AVX2 kernels. All of my kernels use `float64`, so a 256-bit AVX2 vector instruction can process four double-precision values at a time.

Of course, loop unrolling and vectorization are two different things. Unrolling exposes more independent operations, while vector instructions let us process multiple values in one instruction.

## Matrix Multiplication: The Beast

Since the beginning of time, people have imagined AI as a long chain of:

```cpp
if (...) {}
else if (...) {}
...
else {}
```

But with the rise of deep learning and transformer models, I can confidently say that matrix multiplication is everywhere.

If Sifu were here, he'd say, "Look, matrix multiplication is everywhere!"

So let's dig into what we can do with `matmul()`.

The basic implementation we learned during our school and college days looks like this:

```cpp
for (unsigned i = 0; i < m; i++) {
    for (unsigned j = 0; j < n; j++) {
        C[i][j] = 0;

        for (unsigned k = 0; k < l; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

But this implementation can have a significant issue with its memory access pattern.

Look closely at matrix `B`. In the innermost loop, we're moving down a column, jumping `n` elements between consecutive accesses in row-major storage.

Depending on the matrix dimensions and cache behavior, this can lead to inefficient cache utilization. Can we do better? Yes, we can.

```cpp
// Initialize C to zero before these loops.

for (unsigned i = 0; i < m; i++) {
    for (unsigned j = 0; j < l; j++) {
        for (unsigned k = 0; k < n; k++) {
            C[i][k] += A[i][j] * B[j][k];
        }
    }
}
```

Here, we interchange the loops. The innermost loop accesses consecutive elements in a row of `B` and a row of `C`, improving their memory access patterns.

One constraint: `C` must be initialized to zero before accumulating the results.

Can we do better?

Yes, we can.

But that would take us much deeper into cache blocking and tiled matrix multiplication for GPUs. I'll keep that discussion on hold for future articles.

## Broadcasting

I've got a question. Let's say:

```cpp
tensor A(48, 12);
tensor B(48);

tensor C = A.add(B);
```

What should the resulting dimensions be?

Obviously, `C(48, 12)`.

But wait. If you've used NumPy, PyTorch, or TensorFlow, you might be wondering how this works. Under their conventional shape interpretation, `(48, 12)` and `(48,)` aren't compatible for broadcasting.

Remember, my framework uses its own dimension convention. Here, the 48-element tensor aligns with the first dimension of `A`, so its values are reused across the second dimension.

That's broadcasting: allowing tensors with compatible shapes to participate in an operation without explicitly copying the smaller tensor to match the larger one.

Think about a dense layer. The matrix multiplication produces an output for a batch of samples, and then we add the bias to each sample. We don't want to manually duplicate the bias vector for every sample. Broadcasting lets us express that operation naturally.

Sounds simple, right?

The tricky part is determining whether two shapes are compatible, which axes need broadcasting, and how to map each output element to the correct input elements.

A normal element-wise addition can access corresponding elements directly. A broadcasted addition may need to reuse the same input element multiple times.

Understanding broadcasting was crucial for my dense layer implementation. It wasn't just another mathematical operation to add to the API; it was a shape-handling problem that the framework needed to solve correctly.

And trust me, getting the rules right matters a lot when we start composing more complex layers.

## Greater Than Zero — Seriously?

This operation checks every element of a tensor and returns `1` if the element is greater than zero, otherwise `0`.

Something like:

```text
input:  [-2.5, 0.0, 3.7, -1.2]
output: [ 0.0, 0.0, 1.0,  0.0]
```

Why do we need such an obvious operation?

Because neural networks are full of seemingly simple operations that become useful when applied to entire tensors.

Take ReLU, for example:

\[
\operatorname{ReLU}(x)=\max(0,x)
\]

During the forward pass, negative values become zero while positive values pass through unchanged.

During backpropagation, we need the derivative of ReLU. Away from zero, it is:

\[
\operatorname{ReLU}'(x)=
\begin{cases}
0 & x<0\\
1 & x>0
\end{cases}
\]

A greater-than-zero mask can therefore help us propagate gradients through the activation.

Of course, the derivative at exactly zero requires a convention, and the mask alone isn't the complete backward operation.

So, yes, an operation that checks whether a number is greater than zero has a legitimate place in a deep learning framework.

I'll get into how these operations connect to automatic differentiation when we reach graph execution.

For now, let's just say that the little guys matter too. He he.

## Kernel Dispatcher — Who Does the Actual Work?

So, we've got our operations. We've got `add()`, `mul()`, `matmul()`, broadcasting, and a bunch of other operations.

But here's the question: if we have multiple implementations of the same operation, who decides which one to execute?

Enter the kernel dispatcher.

Remember, the mathematical operation and its implementation are two different things. Adding two tensors is a mathematical operation. Whether that addition is performed by a scalar CPU kernel, an AVX2 implementation, or a GPU kernel is an execution decision.

And we don't want every operation to reinvent that decision.

Imagine having to write the same backend-selection logic inside `add()`, `mul()`, `div()`, `matmul()`, and every other operation. That's a lot of duplicated logic, and maintaining it would become a nightmare as the framework grows.

Instead, we separate the operation from the mechanism that selects its implementation.

Conceptually, the flow looks like this:

```text
Tensor operation
       |
       v
Kernel dispatcher
       |
       v
Select a suitable implementation
       |
       +----> Scalar CPU
       |
       +----> AVX2
       |
       +----> CUDA
```

The dispatcher needs to consider the requested execution backend and whether a suitable implementation of that particular operation is available. If the requested kernel isn't available, the framework can use a supported fallback where appropriate.

For example, requesting a GPU implementation doesn't magically mean every operation has a CUDA kernel ready to execute.

And that's the point: the operation describes *what we want to compute*, while the dispatcher helps determine *which implementation will compute it*.

This separation also gives us a natural place to extend the framework when we add new operations or execution backends.

For now, we'll leave the actual kernel implementations for another series. There's plenty of fun to be had there.

*[Insert screenshots of your operation tests and validation results here.]*

To be continued...