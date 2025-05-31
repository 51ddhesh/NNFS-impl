[Previous Session](Session_3.md)
## The Dense Layer Class

[Notebook](/code-files/4_Dense_Layer.ipynb)

Generally, the input for neural networks is not linear.
If such was the case, a simple regression model would perform better. 

![Non Linear Data](/images/spiral_data.png)

<br>To create this data, we use the <code>nnfs</code> library for <code>Python</code>.

## Implementing the Dense Layer Class
In previous sessions, the forward pass was always found using 
```python
output = np.dot(inputs, weights.T) + bias
```
But from now on, to avoid using the transpose function, we will represent the weight matrix as:
```math
weights = \begin{bmatrix} w_{11} & ..w_{1n} \\ w_{mi} & ..w_{mn}\end{bmatrix}_{inputs \times neurons}
```

Where:

![Dense Layer](/images/dense_layer.jpg)

Now, we implement a dense layer class in Python
```python
class Dense_Layer():
    def __init__(self, inputs, neurons):
        self.weights = np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
```
We create the dataset from <code>nnfs.datasets.spiral_data</code>
```python
import nnfs
from nnfs.datasets import spiral_data

X, y = spiral_data(samples=100, classes=3)
```
This creates a sample data with 100 rows and two inputs <code>X</code> and <code>y</code> and 3 features (neurons).
<br>To test, an object `dense_1` is created. `dense_1` has 2 inputs and 3 features/neurons.
```python
dense_1 = Dense_Layer(2, 3)
dense_1.forward_pass(X) 
```
The `forward_pass()` function finds the forward pass and stores it in the `output` (type `numpy.ndarray`).

```python
print(dense_1.output[:5])
```
Printing a sample output:
```python
[[ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
 [ 1.3520580e-04  1.8173116e-05 -1.7020700e-04]
 [ 2.3245417e-04 -2.2105001e-04 -2.2379705e-04]
 [ 3.8226307e-04 -2.8677558e-04 -3.8896195e-04]
 [ 5.7436468e-04 -7.9355465e-05 -6.8033929e-04]]
```
The first column is the output from the first neuron, the second column is the output from the second neuron and the third column is the output from the third neuron in the layer.
