# Loss Functions

[Previous Session](/docs/Sessions/Session_5.md)

[Notebook](/code-files/6_Loss_Functions.ipynb)

## Definition
A neural network makes a prediction based on the data it has been trained on. A loss function, also called a cost or an objective function, is a mathematical function that measures the difference between the network's predicted output and the true target value.

## Need
To learn from data, the network needs feedback—a way to evaluate how good or bad its predictions are. A loss function compares the predicted output to the actual label and returns a single number: the loss.
- High loss → the prediction is far from the true value (poor performance)
- Low loss → the prediction is close to the true value (good performance)

This feedback guides the model during training to improve its accuracy.


## Categorical Cross-Entropy Loss
In this project, we are working on a classification task, so we use the categorical cross-entropy loss. 

Categorical Cross Entropy Loss is defined as:

```math
L = -\sum_{i = 1}^{C} y_{i} \log{(\hat{y}_{i})}
```
More formally, categorical cross-entropy is defined as the negative sum over all classes of the product of the true label and the natural logarithm of the predicted probability.


Say for our classification task, we are working with data that looks like this:

![dataset](/images/spiral_data.png)

The goal is to classify which color a given input belongs to (e.g., Red, Green, or Blue).

- The final layer, using the Softmax activation function, might output:

```python
[0.7 0.1 0.2]
```
This represents the model's predicted probabilities for the Red, Green, and Blue classes respectively.

- The actual label might be:

```python
[1 0 0]
```
This is a one-hot encoded vector indicating that the correct class is Red. `1` indicating `True` and `0` indicating `False`.

For the above values:
```python
TrueLabel = [1 0 0]
```
and

```python
predicted = [0.7 0.1 0.2]
```

The loss is:

```math
L = -\sum_{i = 1}^{3}y_{i}\log{(\hat{y}_{i})} 
```

```math
L = -(1 \cdot \log{(0.7)} + 0 \cdot \log{(0.1)} + 0 \cdot \log{(0.2)})
```

```math
L = -\log{(0.7)}
```


```math
L = 0.3566749439 \approx 0.3567
```
Since only one element of the one-hot vector is non-zero, the loss depends solely on the predicted probability for the correct class label. Thus, the loss simplifies to the negative logarithm of the predicted probability for the true class.

Or simply:
It can be deduced that, to find the loss, only the predicted probability and the index of the true label is required. After that the loss can be found as the negative logarithm of the `probability[true label]`.


The loss follows a negative logarithmic curve and comes closer to zero as the prediction approaches 1. 

## Finding the Loss
### Case 1 - Class Targets are given

Let the batch output from the `Softmax` activation be
```python
softmaxOutputs = [
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
]
```

And the `true_labels` are given as: `[Red, Green, Green]` in the form of:

```python
classTargets = [0 1 1]
```
i.e, the required indices to choose are given in the `classTargets` array.

> 0 value at an index corresponds to the data belonging to the Red colour

> 1 value at an index corresponds to the data belonging to the Green colour

> 2 value at an index corresponds to the data belonging to the Blue colour

Mathematically, the loss is found simply by ignoring the other indices as their value is 0 due to the `one-hot encoding` of `true_labels`.

```math
L_{1} = -\log{(0.7)} \approx 0.3567
```
```math
L_{2} = -\log{(0.5)} \approx 0.6931
```
```math
L_{3} = -\log{(0.9)} \approx 0.1054
```

That means, to find the loss for the above data, we require only:
- The first index (0) from the first row
- The second index (1) from the second row
- The second index (1) from the third row


To implement this in `Python`, we use some of its advanced indexing features


```python
required_values = softmaxOutputs[[0, 1, 2], classTargets]
```
could also be written as:
```python
softmaxOutputs[[0, 1, 2], [0, 1, 1]]
```

This means that:
- From the 0th row of `softmaxOutputs` choose the 0th index
- From the 1st row of `softmaxOutputs` choose the 1st index
- From the 2nd row of `softmaxOutputs` choose the 1st index


```math
softmaxOutpus = \begin{bmatrix} \begin{bmatrix} 0.7 & 0.1 & 0.2 \end{bmatrix} \\ \begin{bmatrix} 0.1 & 0.5 & 0.4 \end{bmatrix} \\ \begin{bmatrix} 0.02 & 0.9 & 0.08 \end{bmatrix} \end{bmatrix}
```

Doing the above turns it to:

```math
softmaxOutputs = \begin{bmatrix} 0.7 & 0.5 & 0.9 \end{bmatrix}
```

Now the logarithm of the above can be found easily to find the loss.

It can be coded out in `Python` quite naively as:

```python
import numpy as np

softmaxOutputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])

classTargets = np.array(
    [0, 1, 1]
)

required = softmaxOutputs[[0, 1, 2], classTargets]

loss = -np.log(required)

print(required)
print(loss)

```
Output:

```python
[0.7 0.5 0.9]
[0.35667494 0.69314718 0.10536052]
```

A more efficient way is covered in the [Notebook](/code-files/6_Loss_Functions.ipynb).


### Case 2 - One Hot Encoding

Here, the `softmaxOutputs` are same as before, but the `true_labels` are in the form of:

```python
import numpy as np

classTargets = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0]
])
```
Here the first column represents the Red colour class, second column represents the Green colour class and the third column represents the Blue colour class.

On placing both the `classTargets` and `softmaxOutputs` side by side, 

```math
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} 0.7 & 0.1 & 0.2 \\ 0.1 & 0.5 & 0.4 \\ 0.02 & 0.9 & 0.08 \end{bmatrix}
```

On performing element wise multiplication, it results in:

```math
\begin{bmatrix} 0.7 & 0 & 0 \\ 0 & 0.5 & 0 \\ 0 & 0.9 & 0\end{bmatrix}
```

Now on executing `numpy.sum(axis=1)`, it results in:
```math
\begin{bmatrix} 0.7 \\ 0.5 \\ 0.9 \end{bmatrix}
```
Here onwards, the usual `-np.log()` is used to find the loss.

Now, to perform element-wise multiplication, we use the regular multiplication sign `*`.

```python
import numpy as np

softmaxOutputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])

classTargets = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0]
])


required = softmaxOutputs * classTargets

print(required)
```

output:

```python
[[0.7 0.  0. ]
 [0.  0.5 0. ]
 [0.  0.9 0. ]]
```

```python
required = np.sum(required, axis=1)
print(required)
```
output

```python
[0.7 0.5 0.9]
```

```python
loss = -np.log(required)
print(f'Loss: {loss}')
print(f'Average Loss: {np.mean(loss)}') 
```
output:

```python
Loss: [0.35667494 0.69314718 0.10536052]
Average Loss: 0.38506088005216804
```

## `Clipping` the Prediction Matrix
Since we are taking a logarithm of the `prediction_matrix`, we must make sure that the values are neither too high nor too low. The curve of the negative natural logarithm (`ln(x)`) shows how quickly the value changes in the range (0, 1).

![negative logarithm](/images/lnx.png)


To make sure that the values always stay in the exclusive range (0, 1), we use the `numpy.clip` function.

```python
clipped = np.clip(predicton, 1e-7, 1 - 1e7)
```
This transforms the `numpy.ndarray` as:
- All values smaller than `1e-7` will be set to `1e-7`
- All values greater than `1 - 1e-7` will be set to `1 - 1e-7`

The Categorical Cross-Entropy Loss is implemented as a `class` in the [Notebook](/code-files/6_Loss_Functions.ipynb)


[Next Session](/docs/Sessions/Session_7.md)
