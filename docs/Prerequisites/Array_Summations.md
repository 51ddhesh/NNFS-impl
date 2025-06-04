# Array Summations and Broadcasting in Python

## Array Summations using `numpy.sum()`

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}_{3 \times 3}
```

```python
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(np.sum(A))
```
The output for above is 
```python
45
```


Therefore, `np.sum(A)` adds each element of the the `numpy.array()`


### Adding `axis` to `numpy.sum()`

```python
print(np.sum(A, axis=None))
```
This results in:
```python
45
```
Therefore, not using any `axis` adds up all the elements of the `numpy.array()`

- `axis = 0`
```python
print(np.sum(A, axis=0))
```
Output
```python
[12 15 18]
```
`axis = 0` sums down the rows of the matrix → across rows → **column-wise sum**
<br>
<br>

```math
sum = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} + \begin{bmatrix} 4 & 5 & 6 \end{bmatrix} + \begin{bmatrix} 7 & 8 & 9 \end{bmatrix} = \begin{bmatrix} 12 & 15 & 18 \end{bmatrix}
```
<br>
<br>

- `axis = 1`
```python
print(np.sum(A, axis=1))
```
Output
```python
[6 15 24]
```
`axis = 1` sums across the columns of the matrix → **row-wise sum**

<br>

```math
sum  \hspace{1mm} across \hspace{1mm} the \hspace{1mm} columns \hspace{1mm} (axis = 1) = \begin{bmatrix} 1 \\ 4 \\ 7 \end{bmatrix} + \begin{bmatrix} 2 \\ 5 \\ 8 \end{bmatrix} + \begin{bmatrix} 3 \\ 6 \\ 9 \end{bmatrix} = \begin{bmatrix} 6 \\ 15 \\ 24 \end{bmatrix}
```

Now, a thing to notice is that, with only `axis` as an argument, the result is a **1 dimensional matrix**. This can be verified as;
<br>

```python
print(np.sum(A, axis=0).shape)
print(np.sum(A, axis=1).shape)
```
Output
```python
(3,)
(3,)
```

### Adding `keepdims` to `numpy.sum()`
`keepdims` maintains the original dimensions of the input `numpy.array()`
<br>This means that, keeping `keepdims = True` results in:
<br>

```python
horizontal = np.sum(A, axis=0, keepdims=True)
vertical = np.sum(A, axis=1, keepdims=True)

print(f'horizontal:\n{horizontal}\n')
print(f'horizontal.shape:\n{horizontal.shape}\n')
print(f'vertical:\n{vertical}\n')
print(f'vertical.shape:\n{vertical.shape}\n')
```
Output
```python
horizontal:
[[12 15 18]]

horizontal.shape:
(1, 3)

vertical:
[[6]
 [15]
 [24]]

vertical.shape:
(3, 1)
```
<br>

```math
horizontal = \begin{bmatrix} \begin{bmatrix} 12 & 15 & 18 \end{bmatrix} \end{bmatrix}_{1 \times 3}
```
<br>

```math
vertical = \begin{bmatrix} \begin{bmatrix} 6 \end{bmatrix} \\ \begin{bmatrix} 15 \end{bmatrix} \\ \begin{bmatrix} 24 \end{bmatrix} \end{bmatrix}_{3 \times 1}
```
<br>

## Broadcasting Rules in Python
From [Session 2](/docs/Sessions/Session_2.md),

```python
output = np.dot(input, weights.T) + bias
```

Here, the `np.dot(input, weights.T)` did not necessarily have the same dimesnsions as `bias`. Below, it is shown how `NumPy` handles this internally.

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}_{3 \times 3}
```
<br>

```math
B = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}_{3 \times 1}
```
<br>

When performing operations on two arrays, `NumPy` compares their shape element-wise from right to left. The dimensions are compatible when:
1. They are equal, or
2. One of them is 1 

### How Broadcasting works
Given two arrays, `NumPy`:
- Aligns the shapes from right to left
- Prepends the dimensions of size 1 or the smaller shape, if needed
- Checks each dimension

Examples:
1. Compatible shape

```python
A.shape = (3, 3)
B.shape = (3, 1)
```
Align right to left:

```Makefile
A:  3 x 3
B:  3 x 1
```
- Dimension 1: 3 and 1 → can be broadcasted
- Dimension 0: 3 and 3 → can be broadcasted

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}_{3 \times 3}
```

```math
B = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}_{3 \times 1}
```

`NumPy` converts `B` as below

```math
B = \begin{bmatrix} 1 & 1 & 1 \\ 2 & 2 & 2 \\ 3 & 3 & 3 \end{bmatrix}_{3 \times 3}
```

`NumPy` then performs normal addition as:

```math
A + B = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}_{3 \times 3} + \begin{bmatrix} 1 & 1 & 1 \\ 2 & 2 & 2 \\ 3 & 3 & 3 \end{bmatrix}_{3 \times 3}
```

<br>

```math
A + B = \begin{bmatrix} 2 & 3 & 4 \\ 6 & 7 & 8 \\ 10 & 11 & 12 \end{bmatrix}
```

Coded out:

```python
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

B = np.array([
    [1], 
    [2],
    [3]
])

C = A + B

print(C)
```

output

```python
[[ 2  3  4]
 [ 6  7  8]
 [10 11 12]]
```


2. Not compatible

```python
A.shape = (3, 2)
B.shape = (3,)
```

Align right to left

```Makefile
A:  3 x 2
B:  1 x 3
```

- Dimension 1: 3 and 2 → not equal and neither is one. Hence `A` and `B` are cannot be broadcasted

### Some Other Examples

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}_{3 \times 3}
```
<br>

```math
B = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}_{1 \times 3}
```

```python
A.shape = (3, 3)
B.shape = (3,)
```

`NumPy` replaces the 'nothing' (no dimension) by 1

```Makefile
A:  3 x 3
B:  1 x 3 
```

`NumPy` convert `B` to:

```math
B = \begin{bmatrix} 1 & 2 & 3 \\ 1 & 2 & 3 \\ 1 & 2 & 3 \end{bmatrix}_{3 \times 3}
```

The final operation performed is:

```python
C = A + B
```

```math
C = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}_{3 \times 3} + \begin{bmatrix} 1 & 2 & 3 \\ 1 & 2 & 3 \\ 1 & 2 & 3 \end{bmatrix}_{3 \times 3}
```

Therefore,

```math 
C = \begin{bmatrix} 2 & 4 & 6 \\ 5 & 7 & 9 \\ 8 & 10 & 12 \end{bmatrix}_{3 \times 3}
```

Coded out:
```python
import numpy as np

A = np.array([
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9]
])

B = np.array([1, 2, 3])

C = A + B

print(C)
```
output

```python   
[[ 2,  4,  6],
 [ 5,  7,  9],
 [ 8, 10, 12]]
```

### Applications

Consider the matrix A

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}_{3 \times 3}
```
The task is to subtract all elements by the maximum value in its respective row.

> In the first row, all elements will be subtracted by 3

> In the second row, all elements will be subtracted by 6

> In the third row, all elements will be subtracted by 9

```math
expected \hspace{1mm} output = \begin{bmatrix} -2 & -1 & 0 \\ -2 & -1 & 0 \\ -2 & -1 & 0 \end{bmatrix}_{3 \times 3}
```


- Approach 1

```python
MaxFromRow = np.max(A, axis=1)
print(MaxFromRow)
print(MaxFromRow.shape)
```

output

```python
[3 6 9]
(3,)
```
This creates a `1D array` which when subtracted from `A` results in:

```python
B = A - MaxFromRow
print(B)
```
output

```python
[[-2 -4 -6]
 [ 1 -1 -3]
 [ 4  2  0]]
```
This happens because, `NumPy` converts the `1D array` from:

```math
MaxFromRow = \begin{bmatrix} 3 & 6 & 9 \end{bmatrix}
```
to a `2D array` as

```math
MaxFromRow = \begin{bmatrix} 3 & 6 & 9 \\ 3 & 6 & 9 \\ 3 & 6 & 9 \end{bmatrix}_{3 \times 3}
```

- Approach 2

```python
MaxFromRow = np.max(A, axis=1, keepdims=True)
print(MaxFromRow)
print(MaxFromRow.shape)
```

output

```python
[[3]
 [6]
 [9]]
(3, 1)
```

Now this results in a `2D array` which can be used to perform the required task

```python
B = A - MaxFromRow
print(B)
```
output

```python
[[-2 -1  0]
 [-2 -1  0]
 [-2 -1  0]]
```

This happens because the `MaxFromRow` is a `2D array` as `keepdims` was set to `True`

```math
MaxFromRow = \begin{bmatrix} 3 \\ 6 \\ 9 \end{bmatrix}_{3 \times 1}
```
Which `NumPy` converts to 


```math
MaxFromRow = \begin{bmatrix} 3 & 3 & 3 \\ 6 & 6 & 6 \\ 9 & 9 & 9 \end{bmatrix}_{3 \times 3}
```
