[Previous Session](Lecture_3)
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
