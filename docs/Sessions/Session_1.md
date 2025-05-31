# The Neuron 

[Notebook](/code-files/1_Neurons_and_Layers.ipynb)

## Implementing the neuron

![A neuron](/images/neuron.jpg)

In Machine Learning, the Neuron is the basic computational unit of a neural network. A neural network is designed to imitate the brain. Inspired by the biological neurons, it receives input values, which are multiplied by weights and a bias term is added to get the final output.

```math
	output = \sum (input \times weights) + bias
```
In `Python`, it can be implemented as below, also demonstrated in the [notebook](/code-files/1_Neurons_and_Layers.ipynb).

```python
inputs = []
weights = []
bias

output = 0
for i in range(0, len(weights)):
	output += input[i] * weights[i]

output += bias
```

## Implementing a layer of neurons

![A Layer of neurons](/images/layer_of_neurons.jpg)

A layer is a collection of neurons that all receive the same inputs but operate with different weights and biases. Each neuron in the layer processes the input independently and produces its own output. The collection of outputs from all neurons in the layer forms the layer's output.

```python
inputs = []
weights = [[], [], [], [] ...]
biases = []

output = [0] * len(weights)

for i in range(0, len(weights)):
	for j in range(0, len(inputs)):
		outputs[i] = inputs[j] * weights[i][j]

for i in range(0, len(biases)):
	outputs[i] += biases[i]
```

[Next Session](./Session_2.md)
