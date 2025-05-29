[Previous Session](Session_2.md)

## Implementing Multiple Layers

[Notebook](/code-files/3_Multiple_Layers.ipynb)

In the previous section, we calculated the output of a single neuron and a single layer of neuron using numpy.dot

```python
outputs = np.dot(inputs, weights.T) + biases
```

#### Multi Layer Forward Pass (Deep Neural Networks)

```python
inputs = np.array([])
weights1 = np.array([])
weights2 = np.array([])

biases1 = np.array([])
biases2 = np.array([])
```

The outputs from a layer are used as inputs for the next layer

```python
layer1_output = np.dot(inputs, weights1.T) + biases1
```

```python
layer2_output = np.dot(layer1_output, weights2.T) + biases2
```

For testing purposes, the layers can be created at random

```python
import numpy as np

inputs = np.random.rand(3, 4) # 3 samples, 4 features
w1 = np.random.rand(3, 4) # 3 samples, 4 inputs
w2 = np.random.rand(3, 3) # 3 samples, 3 inputs
w3 = np.random.rand(3, 3) # similar as above
w4 = np.random.rand(3, 3)
```

Similarly, the biases are also created at random

```python
b1 = np.random.rand(3)
b2 = np.random.rand(3)
b3 = np.random.rand(3)
b4 = np.random.rand(3)
```

Now the outputs can be found as:

```python
l1 = np.dot(inputs, w1.T) + b1
```

```python
l2 = np.dot(l1, w2.T) + b2
```

```python
l3 = np.dot(l2, w3.T) + b3
```

```python
l4 = np.dot(l3, w4.T) + b4
```

Each layer uses the output of the previous layer as input

[Next Session](Session_4.md)
