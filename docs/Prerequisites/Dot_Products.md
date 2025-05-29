The Dot Product is a function from NumPy
`np.dot(a, b)`

## Dot Products between two vectors

```python
a = np.array([a1, a2, a3])
b = np.array([b1, b2, b3])
c = np.dot(a, b) # same as np.dot(b, a)
```

The formula for dot product between two vectors is 
```math
\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + a_3 b_3
```

## Dot Products between a vector and a matrix

```python
a = [1, 2, 3] # a.shape = (3,) -> vector 
b = [[4, 5, 6], # b.shape = (3, 3) -> matrix
	 [7, 8, 9],
	 [10, 11, 12]]

```

#### <code>np.dot(a, b)</code> - vector  * matrix

```math
a = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}_{1\times3}
```

```math
b = \begin{bmatrix} 4 & 5 & 6 \\ 7 & 8 & 9 \\ 10 & 11 & 12\end {bmatrix}_{3\times3} 
```

```math
a_1 = \begin{bmatrix} 1 & 2 & 3\end{bmatrix}_{1\times3}
```

```math
b_i \hspace{2 mm} is \hspace{2 mm} the \hspace{2 mm} i^{th} \hspace{2 mm} column \hspace{2 mm} of \hspace{2 mm} the \hspace{2 mm} matrix \hspace{2 mm} b
```

```math
b_1 = \begin{bmatrix} 4 & 7 & 10 \end{bmatrix}_{1\times3}
```

```math
b_2 = \begin{bmatrix} 5 & 8 & 11 \end{bmatrix}_{1\times3}
```

```math
b_3 = \begin{bmatrix} 6 & 9 & 12 \end{bmatrix}_{1\times3}
```

```math
	a \cdot b = 
	\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}_{1\times3} \cdot \begin{bmatrix} 4 & 5 & 6 \\ 7 & 8 & 9 \\ 10 & 11 & 12 \end{bmatrix}_{3\times3}  
```

```math
= \begin{bmatrix} \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \cdot \begin{bmatrix} 4 & 7 & 10 \end{bmatrix} \hspace{3mm} \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \cdot \begin{bmatrix} 5 & 8 & 11 \end{bmatrix} \hspace{3mm} \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \cdot \begin{bmatrix} 6 & 9 & 12 \end{bmatrix}\end{bmatrix}
```

```math
	a \cdot b = \begin{bmatrix} 48 & 54 & 60 \end{bmatrix}_{1\times3}
```

Therefore 

```python
np.dot(a, b) = [np.dot(a, col1), np.dot(a, col2), np.dot(a, col3)]
# Row Matrix
```

Hence, <code>np.dot(vector, matrix)</code> is the same as computing the **`dot product`** of `a` with **`each column` of `b`**
#### <code>np.dot(b, a)</code> - matrix * vector

```math
b = \begin{bmatrix} 4 & 5 & 6 \\ 7 & 8 & 9 \\ 10 & 11 & 12 \end{bmatrix}_{3\times3}
```

```math
a = \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}_{1\times3}
```

```math
b_i \hspace{2mm} is \hspace{2mm} the \hspace{2mm} i^{th} \hspace{2mm} row \hspace{2mm} of \hspace{2mm} the \hspace{2mm} matrix \hspace{2mm} b
```

```math
b_1 = \begin{bmatrix} 4 & 5 & 6\end{bmatrix}_{1\times3}
```

```math
b_2 = \begin{bmatrix} 7 & 8 & 9 \end{bmatrix}_{1\times3}
```
```math
b_3 = \begin{bmatrix} 10 & 11 & 12 \end{bmatrix}_{1\times3}
```

```math
b \cdot a = 
\begin{bmatrix} 4 & 5 & 6 \\ 7 & 8 & 9 \\ 10 & 11 & 12 \end{bmatrix}_{3\times3} \cdot \begin{bmatrix} 1 \\ 2 \\ 3\end{bmatrix}_{3\times1} = \begin{bmatrix} \begin{bmatrix} 4 & 5 & 6 \end{bmatrix}\cdot \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \\ \begin{bmatrix} 7 & 8 & 9 \end{bmatrix}\cdot \begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \\ \begin{bmatrix} 10 & 11 & 12 \end{bmatrix}\cdot \begin{bmatrix} 1 & 2 & 3 \end{bmatrix}  \end{bmatrix} = \begin{bmatrix}4\cdot1 + 5\cdot2 + 6\cdot3 \\ 7\cdot1 + 8\cdot2 + 9\cdot3 \\ 10\cdot1 + 11\cdot2 + 12\cdot3 \end{bmatrix} = \begin{bmatrix} 32 \\ 50 \\ 68 \end{bmatrix}_{3\times1}
```

Therefore
```python
np.dot(b, a) = [
	np.dot(b1, a),
	np.dot(b2, a),
	np.dot(b3, a)
]

# Column Matrix
```

Hence, <code>np.dot(matrix, vector)</code> is the same as computing the **`dot product`** of `a` with **`each row` of `b`**


## Dot Products between two matrices

```math
	a = \begin{bmatrix} a_{11} & a_{12} & a_{13} & a_{14} \\ a_{21} & a_{22} & a_{23} & a_{24} \\ a_{31} & a_{32} & a_{33} & a_{34} \end{bmatrix}_{3\times4} \hspace{5mm} b = \begin{bmatrix} b_{11} & b_{12} & b_{13} \\ b_{21} & b_{22} & b_{23} \\ b_{31} & b_{32} & b_{33} \\ b_{41} & b_{42} & b_{43} \end{bmatrix}_{4\times3}
```

The matrices are divided as:
$$ r_1 = \begin{bmatrix} a_{11} & a_{12} & a_{13} & a_{14}\end{bmatrix}_{1\times3} $$
$$ r_2 = \begin{bmatrix} a_{21} & a_{22} & a_{23} & a_{24} \end{bmatrix}_{1\times3} $$
$$ r_3 = \begin{bmatrix} a_{31} & a_{32} & a_{33} & a_{34} \end{bmatrix}_{1\times3} $$
And:
$$ c_1 = \begin{bmatrix} b_{11} & b_{21} & b_{31} & b_{41}\end{bmatrix}_{1\times3} $$
$$ c_2 = \begin{bmatrix} b_{12} & b_{22} & b_{32} & b_{42}   \end{bmatrix}_{1\times3} $$
$$ c_3 = \begin{bmatrix} b_{13} & b_{23} & b_{33} & b_{43} \end{bmatrix}_{1\times3} $$
Then, 
```math
a\cdot b = \begin{bmatrix} r_1\cdot c_1 & r_1\cdot c_2 & r_1\cdot c_3 \\ r_2\cdot c_1 & r_2\cdot c_2 & r_2 \cdot c_3 \\ r_3\cdot c_1 & r_3\cdot c_2 & r_3 \cdot c_3 \end{bmatrix}_{3\times3} 
```

Python treats this as a normal multiplication between matrices
