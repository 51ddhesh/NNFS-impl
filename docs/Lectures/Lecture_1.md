Implemented the neuron and a layer of neurons 

#### The Neuron

```python
inputs = []
weights = []
bias

output = 0
for i in range(0, len(weights)):
	output += input[i] * weights[i]

output += bias
```

#### Layer

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

[Next Lecture](Lecture_2)
