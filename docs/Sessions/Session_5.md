# Activation Functions 

[Previous Session](./Session_4.md)

[Prerequisite - Broadcasting Rules](/docs/Prerequisites/Array_Summations.md)

[Notebook](/code-files/5_Activation_Functions.ipynb)



## Activation Functions

### Definition of an Activation Function

An activation function is a mathematical function used in artificial neural networks to determine the output of a neuron based on its input.

>  Input → Weighted Sum → Activation Function → Output


```math
output = \phi(w_{1}x_{1} + w_{2}x_{2} + \cdot \cdot \cdot + w_{n}x_{n} + b)
```

Where `x` are the inputs, `w` are the weights, `b` is the bias and Φ (phi) is the activation function.


![Activation Function in a layer of neurons](/images/activation_1.jpg)

Below is what a network looks like with activation functions and multiple layers in a block representation

![Block Representation](/images/activation_block.jpg)


### Need of an Activation Function
The most important reason to use an activation function, is to implement non linearities.
<br>Without activation functions, a neural network would just be a stack of linear equations. No matter how many layers you add, it could only learn linear patterns. Activation functions let it learn complex, non-linear relationships.


![Need of Activation Functions](/images/need_of_activation.png)
- The red coloured curve is the underlying data
- The blue coloured line shows a network with a linear activation.
- The green coloured curve shows a network with a `ReLU` activation.


### Different Activation Functions
#### Rectified Linear Unit (`ReLU`) Activation
Defined as:

```math
y = max(0, x)
```
Plotted as:

![ReLU](/images/relu.png)

It is particularly useful in capturing rather complex shapes and functions. [Watch a `ReLU` function capturing a sinusoidal curve](https://nnfs.io/mvp/).

in `Python`
```python
import numpy as np

output = np.maximum(0, input)
```

#### Softmax Activation

`Softmax` is a multi-class classification function which is used widely for the data that we are dealing with. `Softmax` makes sure that the output is between (0, 1) - which makes it perfect for using as a probabilistic output.

Consider we have three outputs from the final layer as - 5, 10 and 15.

Softmax converts this outputs as:

```math
output_{1} = \frac{e^{5}}{e^{5} + e^{10} + e^{15}} 
```

```math
output_{2} = \frac{e^{10}}{e^{5} + e^{10} + e^{15}}
```

```math
output_{3} = \frac{e^{15}}{e^{5} + e^{10} + e^{15}}
```

Coded in `Python` as:

```python
import numpy as np

inputs = np.array([])
exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
```

`np.max(inputs, axis=1, keepdims=True)` gets the maximum value in each row and stores it in a column matrix of same number of rows as the `inputs` `numpy.ndarray` and a single column.<br>
Since, the input can have values very large in magnitude, we perform the above minimization of values so that the exponential values do not go out of limits. This is done by subtracting the maximum value in a row from all the elements in a row. We then store it in `exponential_values` which finds the exponent of the minimized values as well.

```python
probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True) 
```

Example runthrough:

```python
import numpy as np
a = np.array([
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2],
    [-1.5, 2.7, 3.3, -0.8]
])

maxFromRow = np.max(a, axis=1, keepdims=True).shape
print(maxFromRow)
```

output:

```python
[[3. ]
 [5. ]
 [3.3]]
```

```python
a = a - np.max(a, axis=1, keepdims=True)
print(a)
```

output

```python
[[-2.  -1.   0.  -0.5]
 [-3.   0.  -6.  -3. ]
 [-4.8 -0.6  0.  -4.1]]
```

```python
exponential = np.exp(a)
print(exponential)
```
output
```python
[[0.13533528 0.36787944 1.         0.60653066]
 [0.04978707 1.         0.00247875 0.04978707]
 [0.00822975 0.54881164 1.         0.01657268]]
```

```python
probabilities = exponential / np.sum(exponential, axis=1, keepdims=True)
print(probabilities)
```

output

```python
[[0.06414769 0.17437149 0.47399085 0.28748998]
 [0.04517666 0.90739747 0.00224921 0.04517666]
 [0.00522984 0.34875873 0.63547983 0.0105316 ]]
```

We can verify this result by:

```python
print(np.sum(probabilities, axis=1))
```
output:

```python
[1. 1. 1.]
```




