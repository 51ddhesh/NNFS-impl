[Previous Lecture](Lecture_1)
Implement the neuron, layer, batch using the numpy dot product

Prerequisites: [Dot Product](Dot_Products.md)

### A Neuron using a Dot Product

From [Lecture 1](Lecture_1), 
```python
inputs = []
weights = []

output = inputs[i] * weights[i] + bias
```
This same operation can be done using a dot product
```python
import numpy as np

inputs = np.array([])
weights = np.array([])

output = np.dot(weights, input) + bias
```


### A Layer using a Dot Product

Similar to the neuron, an entire layer can be coded using a single line of dot product

```python
inputs = np.array([1, 2, 3, 2.5])

weights = np.array([
    [0.2, 0.8, -0.5, 1], # weights[0]
    [0.5, -0.91, 0.26, -0.5], # weights[1]
    [-0.26, -0.27, 0.17, 0.87] # weights[2]
])

biases = np.array([2, 3, 0.5])

outputs = np.dot(weights, inputs) + biases # or np.dot(inputs, weights.T)
```

 - The '+' operator is overloaded in the numpy class to add arrays
**An important point to note is that <code>np.dot(weights, inputs)</code> is used**
But it gives the same result as <code>np.dot(inputs, weights.T)</code> where `weights.T` is the transpose of the `weights` matrix.

### Batch Data

A batch data refers to a larger quantity of inputs, in a higher dimension

```python
batch = np.array([
    [1, 2, 3, 2.5],
    [2, 5, -1, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
])

weights = np.array([
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
])

biases = np.array([2.0, 3.0, 0.5])
```

In this case, again, we use `np.dot(batch, weights.T) + biases` to find the forward pass.

$$
batch = \begin{bmatrix} 1 & 2 & 3 & 2.5 \\ 2 & 5 & -1 & 2.0 \\ -1.5 & 2.7 & 3.3 & -0.8 \end{bmatrix}_{3\times4}
$$
$$
weights = \begin{bmatrix} 0.2 & 0.8 & -0.5 & 1 \\ 0.5 & -0.91 & 0.26 & -0.5 \\ -0.26 & -0.27 & 0.17 & 0.87 \end{bmatrix}_{3\times4}
$$
Now the order of both `batch` and `weights` is same, to find the forward pass here, we need:
$$
outputs_i = \begin{bmatrix} weights_i \cdot batch_1 & weights_i\cdot batch_2 & weights_i \cdot batch_3 \end{bmatrix}
$$
Therefore, we take the transpose of `weights` so that the number of columns of `batch` is the same as the number of rows in the transpose of `weights`.

```python
outputs = np.dot(batch, weights.T) + biases
```
Summarizing, 

```
outputs = np.dot(weights, inputs) + bias == np.dot(inputs, weights.T) + bias
```

[Next Lecture](Lecture_3.md)
