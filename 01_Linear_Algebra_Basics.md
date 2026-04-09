# 01. Linear Algebra for Machine Learning

---

## Why Linear Algebra?
Linear Algebra is the backbone of Machine Learning. Every algorithm, from simple linear regression to complex neural networks, uses linear algebra concepts internally.

---

## 📚 Core Concepts

### 1. **Scalars**
- A single number
- Example: `5`, `3.14`, `-2`

```python
# Python example
x = 5  # This is a scalar
```

---

### 2. **Vectors**
- An ordered array of numbers
- Can be thought of as a point in space or direction

**Types:**
- **Row Vector**: `[1, 2, 3]`
- **Column Vector**: 
  ```
  [1]
  [2]
  [3]
  ```

**Python Implementation:**
```python
import numpy as np

# Row vector
v1 = np.array([1, 2, 3])

# Column vector
v2 = np.array([[1],
               [2],
               [3]])

print("Row vector:", v1)
print("Column vector:\n", v2)
```

**Properties:**
- **Length/Magnitude**: Distance from origin
  - Formula: `||v|| = √(v₁² + v₂² + ... + vₙ²)`
  
```python
# Calculate magnitude
v = np.array([3, 4])
magnitude = np.linalg.norm(v)
print(f"Magnitude: {magnitude}")  # Output: 5.0
```

---

### 3. **Matrices**
- A 2D array of numbers
- Has rows and columns
- Shape: `m × n` (m rows, n columns)

**Example:**
```
A = [1  2  3]
    [4  5  6]
```
This is a 2×3 matrix

**Python Implementation:**
```python
import numpy as np

# Create a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Matrix A:")
print(A)
print(f"Shape: {A.shape}")  # Output: (2, 3)
```

**Special Matrices:**

1. **Identity Matrix (I)**: Diagonal elements are 1, rest are 0
   ```
   I = [1  0  0]
       [0  1  0]
       [0  0  1]
   ```
   
   ```python
   I = np.eye(3)  # 3×3 identity matrix
   ```

2. **Zero Matrix**: All elements are 0
   ```python
   Z = np.zeros((2, 3))  # 2×3 zero matrix
   ```

---

### 4. **Matrix Operations**

#### **Addition/Subtraction**
- Matrices must have the same shape
- Add/subtract corresponding elements

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A + B
print("A + B =")
print(C)
# Output:
# [[ 6  8]
#  [10 12]]
```

#### **Scalar Multiplication**
- Multiply every element by a scalar

```python
A = np.array([[1, 2], [3, 4]])
result = 2 * A
print(result)
# Output:
# [[2  4]
#  [6  8]]
```

#### **Transpose**
- Flip rows and columns
- Notation: `Aᵀ`

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_transpose = A.T
print("Original:")
print(A)
print("\nTranspose:")
print(A_transpose)
# Output:
# [[1  4]
#  [2  5]
#  [3  6]]
```

---

### 5. **Dot Product (VERY IMPORTANT)**

#### **Vector Dot Product**
- Multiply corresponding elements and sum them
- Formula: `a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ`

**Geometric meaning**: Measures how much two vectors point in the same direction

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot_product = np.dot(a, b)
# Same as: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
print(f"Dot product: {dot_product}")
```

**Key Property:**
- If dot product = 0, vectors are perpendicular (orthogonal)

---

#### **Matrix Multiplication**
- Not element-wise!
- Rule: `(m×n) × (n×p) = (m×p)`
- Number of columns in first matrix must equal number of rows in second

**Example:**
```
A = [1  2]    B = [5  6]
    [3  4]        [7  8]

A × B = [1*5+2*7  1*6+2*8]   = [19  22]
        [3*5+4*7  3*6+4*8]     [43  50]
```

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = np.dot(A, B)
# Or use: C = A @ B
print("A × B =")
print(C)
```

**Why is this important in ML?**
- Neural networks use matrix multiplication for forward propagation
- Linear regression: `y = X × w + b`
- Image transformations, data preprocessing, etc.

---

### 6. **Element-wise Operations (Hadamard Product)**
- Multiply corresponding elements
- Symbol: `⊙`

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A * B  # Element-wise multiplication in NumPy
print("A ⊙ B =")
print(C)
# Output:
# [[ 5  12]
#  [21  32]]
```

---

## 🎯 Machine Learning Applications

### 1. **Linear Regression**
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

In matrix form:
y = Xw + b

Where:
- X is the data matrix (features)
- w is the weight vector
- b is bias (scalar)
```

### 2. **Neural Networks**
```
Output = Activation(W × Input + b)

Where:
- W is weight matrix
- Input is feature vector
- b is bias vector
```

### 3. **Image Processing**
- Images are matrices (height × width × channels)
- Filters/kernels use matrix operations

---

## 📝 Practice Exercises

### Exercise 1: Vector Operations
```python
import numpy as np

# Create two vectors
v1 = np.array([2, 3, 4])
v2 = np.array([1, 0, -1])

# Task: Calculate
# 1. Sum of vectors
# 2. Dot product
# 3. Magnitude of v1
# 4. Are they orthogonal?
```

### Exercise 2: Matrix Multiplication
```python
# Given:
A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

# Task: Multiply A × B
# Verify the shape is correct
```

### Exercise 3: Simple Linear Transformation
```python
# Data matrix (3 samples, 2 features)
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Weights
w = np.array([[0.5],
              [0.3]])

# Task: Calculate predictions using y = X × w
```

---

## 🔑 Key Takeaways

1. **Vectors** represent data points or directions
2. **Matrices** represent datasets or transformations
3. **Dot product** is fundamental to ML algorithms
4. **Matrix multiplication** is used in neural networks
5. **NumPy** is your best friend for these operations

---

## 📚 Next Steps

After mastering these basics:
1. Practice with NumPy arrays
2. Understand how data is represented as matrices
3. Move to next topic: Probability & Statistics

---

## 🔗 Resources

- NumPy Documentation: https://numpy.org/doc/
- 3Blue1Brown Linear Algebra Series (YouTube)
- Khan Academy: Linear Algebra

---

**Remember**: You don't need to be a math expert. Focus on understanding:
- What these operations do
- When to use them
- How to implement them in Python
