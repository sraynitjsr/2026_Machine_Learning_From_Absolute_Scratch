"""
Linear Algebra Basics for Machine Learning
Practical Python examples using NumPy
"""

import numpy as np


def section_divider(title):
    """Print a formatted section divider"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def demo_scalars():
    """Demonstrate scalar operations"""
    section_divider("1. SCALARS")

    x = 5
    y = 3.14
    z = -2

    print(f"Scalar x: {x}")
    print(f"Scalar y: {y}")
    print(f"Scalar z: {z}")
    print(f"Operations: x + z = {x + z}, x * y = {x * y}")


def demo_vectors():
    """Demonstrate vector operations"""
    section_divider("2. VECTORS")

    # Row vector
    v1 = np.array([1, 2, 3])
    print("Row vector v1:", v1)
    print(f"Shape: {v1.shape}")

    # Column vector
    v2 = np.array([[1],
                   [2],
                   [3]])
    print("\nColumn vector v2:")
    print(v2)
    print(f"Shape: {v2.shape}")

    # Vector magnitude
    print("\n--- Vector Magnitude ---")
    v = np.array([3, 4])
    magnitude = np.linalg.norm(v)
    print(f"Vector: {v}")
    print(f"Magnitude: {magnitude}")
    print(f"Manual calculation: √(3² + 4²) = √(9 + 16) = √25 = 5.0")

    # Vector addition
    print("\n--- Vector Addition ---")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = a + b
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {c}")

    # Scalar multiplication
    print("\n--- Scalar Multiplication ---")
    scalar = 2
    result = scalar * a
    print(f"{scalar} * {a} = {result}")


def demo_matrices():
    """Demonstrate matrix operations"""
    section_divider("3. MATRICES")

    # Create a matrix
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])

    print("Matrix A:")
    print(A)
    print(f"Shape: {A.shape} (2 rows × 3 columns)")

    # Identity matrix
    print("\n--- Identity Matrix ---")
    I = np.eye(3)
    print("3×3 Identity matrix:")
    print(I)

    # Zero matrix
    print("\n--- Zero Matrix ---")
    Z = np.zeros((2, 3))
    print("2×3 Zero matrix:")
    print(Z)

    # Ones matrix
    print("\n--- Ones Matrix ---")
    O = np.ones((2, 4))
    print("2×4 Ones matrix:")
    print(O)


def demo_matrix_operations():
    """Demonstrate matrix arithmetic operations"""
    section_divider("4. MATRIX OPERATIONS")

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    # Addition
    print("\n--- Addition ---")
    C = A + B
    print("A + B =")
    print(C)

    # Subtraction
    print("\n--- Subtraction ---")
    D = A - B
    print("A - B =")
    print(D)

    # Scalar multiplication
    print("\n--- Scalar Multiplication ---")
    scalar = 2
    E = scalar * A
    print(f"{scalar} * A =")
    print(E)

    # Transpose
    print("\n--- Transpose ---")
    original = np.array([[1, 2, 3],
                        [4, 5, 6]])
    transposed = original.T
    print("Original matrix:")
    print(original)
    print(f"Shape: {original.shape}")
    print("\nTransposed matrix:")
    print(transposed)
    print(f"Shape: {transposed.shape}")


def demo_dot_product():
    """Demonstrate dot product (MOST IMPORTANT)"""
    section_divider("5. DOT PRODUCT (CRITICAL FOR ML)")

    # Vector dot product
    print("--- Vector Dot Product ---")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    dot_product = np.dot(a, b)
    manual_calc = 1*4 + 2*5 + 3*6

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a · b = {dot_product}")
    print(f"Manual: 1*4 + 2*5 + 3*6 = {manual_calc}")

    # Orthogonal vectors
    print("\n--- Orthogonal Vectors ---")
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    dot = np.dot(v1, v2)
    print(f"v1 = {v1}, v2 = {v2}")
    print(f"v1 · v2 = {dot}")
    print("Dot product = 0 means vectors are perpendicular!")


def demo_matrix_multiplication():
    """Demonstrate matrix multiplication"""
    section_divider("6. MATRIX MULTIPLICATION")

    A = np.array([[1, 2],
                  [3, 4]])

    B = np.array([[5, 6],
                  [7, 8]])

    print("Matrix A (2×2):")
    print(A)
    print("\nMatrix B (2×2):")
    print(B)

    # Matrix multiplication using np.dot
    C1 = np.dot(A, B)
    print("\n--- Using np.dot(A, B) ---")
    print("A × B =")
    print(C1)

    # Matrix multiplication using @ operator
    C2 = A @ B
    print("\n--- Using A @ B (same result) ---")
    print("A × B =")
    print(C2)

    # Show calculation step by step
    print("\n--- Manual Calculation ---")
    print("First element [0,0]: 1*5 + 2*7 =", 1*5 + 2*7)
    print("First element [0,1]: 1*6 + 2*8 =", 1*6 + 2*8)
    print("Second element [1,0]: 3*5 + 4*7 =", 3*5 + 4*7)
    print("Second element [1,1]: 3*6 + 4*8 =", 3*6 + 4*8)

    # Different dimensions
    print("\n--- Different Dimensions ---")
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])

    Y = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])

    Z = np.dot(X, Y)
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"X × Y shape: {Z.shape}")
    print("X × Y =")
    print(Z)


def demo_element_wise_operations():
    """Demonstrate element-wise operations"""
    section_divider("7. ELEMENT-WISE OPERATIONS (Hadamard Product)")

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    # Element-wise multiplication
    print("\n--- Element-wise Multiplication (A * B) ---")
    C = A * B
    print("A ⊙ B =")
    print(C)
    print("Note: Each element multiplied individually")

    # Element-wise division
    print("\n--- Element-wise Division (A / B) ---")
    D = A / B
    print("A ÷ B =")
    print(D)

    # Element-wise power
    print("\n--- Element-wise Power (A ** 2) ---")
    E = A ** 2
    print("A² =")
    print(E)


def demo_ml_applications():
    """Demonstrate ML applications of linear algebra"""
    section_divider("8. MACHINE LEARNING APPLICATIONS")

    print("--- Linear Regression Example ---")
    # Data: [bias, feature1, feature2]
    X = np.array([[1, 2, 3],    # Sample 1
                  [1, 4, 5],    # Sample 2
                  [1, 6, 7]])   # Sample 3

    # Weights: [bias_weight, weight1, weight2]
    w = np.array([[0.5],
                  [0.3],
                  [0.2]])

    print("Data matrix X (3 samples, 3 features including bias):")
    print(X)
    print(f"Shape: {X.shape}")

    print("\nWeight vector w:")
    print(w)
    print(f"Shape: {w.shape}")

    # Predictions: y = X × w
    predictions = np.dot(X, w)
    print("\nPredictions (y = X × w):")
    print(predictions)

    print("\n--- Neural Network Layer Example ---")
    # Input layer (1 sample, 4 features)
    input_layer = np.array([[0.5, 0.8, 0.3, 0.9]])

    # Weights (4 input neurons to 3 hidden neurons)
    weights = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6],
                       [0.7, 0.8, 0.9],
                       [0.2, 0.3, 0.4]])

    # Bias for 3 neurons
    bias = np.array([[0.1, 0.1, 0.1]])

    print("Input layer:", input_layer)
    print(f"Shape: {input_layer.shape}")

    print("\nWeights matrix:")
    print(weights)
    print(f"Shape: {weights.shape}")

    # Forward pass: output = input × weights + bias
    output = np.dot(input_layer, weights) + bias
    print("\nOutput (before activation):")
    print(output)
    print(f"Shape: {output.shape}")


def demo_practice_exercises():
    """Practice exercises with solutions"""
    section_divider("9. PRACTICE EXERCISES WITH SOLUTIONS")

    print("--- Exercise 1: Vector Operations ---")
    v1 = np.array([2, 3, 4])
    v2 = np.array([1, 0, -1])

    print(f"v1 = {v1}")
    print(f"v2 = {v2}")

    # Sum
    sum_v = v1 + v2
    print(f"\n1. Sum: v1 + v2 = {sum_v}")

    # Dot product
    dot = np.dot(v1, v2)
    print(f"2. Dot product: v1 · v2 = {dot}")

    # Magnitude
    mag = np.linalg.norm(v1)
    print(f"3. Magnitude of v1: {mag:.4f}")

    # Orthogonal check
    print(f"4. Are they orthogonal? {dot == 0} (dot product = {dot})")

    print("\n--- Exercise 2: Matrix Multiplication ---")
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])

    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])

    print("Matrix A:")
    print(A)
    print(f"Shape: {A.shape}")

    print("\nMatrix B:")
    print(B)
    print(f"Shape: {B.shape}")

    C = np.dot(A, B)
    print("\nA × B =")
    print(C)
    print(f"Result shape: {C.shape}")

    print("\n--- Exercise 3: Linear Transformation ---")
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])

    w = np.array([[0.5],
                  [0.3]])

    print("Data matrix X (3 samples, 2 features):")
    print(X)

    print("\nWeights w:")
    print(w)

    y = np.dot(X, w)
    print("\nPredictions y = X × w:")
    print(y)


def demo_advanced_operations():
    """Demonstrate some advanced operations"""
    section_divider("10. BONUS: ADVANCED OPERATIONS")

    # Determinant
    print("--- Determinant ---")
    A = np.array([[1, 2],
                  [3, 4]])
    det = np.linalg.det(A)
    print("Matrix A:")
    print(A)
    print(f"Determinant: {det}")

    # Inverse
    print("\n--- Matrix Inverse ---")
    A_inv = np.linalg.inv(A)
    print("Inverse of A:")
    print(A_inv)

    # Verify: A × A^(-1) = I
    verify = np.dot(A, A_inv)
    print("\nVerification (A × A^(-1) should be I):")
    print(verify)

    # Eigenvalues and Eigenvectors
    print("\n--- Eigenvalues and Eigenvectors ---")
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("Eigenvalues:")
    print(eigenvalues)
    print("\nEigenvectors:")
    print(eigenvectors)

    # Reshape
    print("\n--- Reshape Operations ---")
    arr = np.array([1, 2, 3, 4, 5, 6])
    reshaped = arr.reshape(2, 3)
    print(f"Original: {arr}")
    print(f"Reshaped to 2×3:")
    print(reshaped)

    # Flatten
    flattened = reshaped.flatten()
    print(f"Flattened back: {flattened}")


def main():
    """Main function to run all demonstrations"""
    print("\n" + "★" * 60)
    print("  LINEAR ALGEBRA FOR MACHINE LEARNING")
    print("  Python Examples with NumPy")
    print("★" * 60)

    # Run all demonstrations
    demo_scalars()
    demo_vectors()
    demo_matrices()
    demo_matrix_operations()
    demo_dot_product()
    demo_matrix_multiplication()
    demo_element_wise_operations()
    demo_ml_applications()
    demo_practice_exercises()
    demo_advanced_operations()

    # Final message
    section_divider("COMPLETE!")
    print("You've completed all Linear Algebra basics!")
    print("\nKey Takeaways:")
    print("1. Vectors and matrices are fundamental data structures")
    print("2. Dot product is crucial for ML algorithms")
    print("3. Matrix multiplication ≠ Element-wise multiplication")
    print("4. NumPy makes all operations simple and efficient")
    print("\nNext steps: Practice these operations and move to")
    print("Probability & Statistics!")
    print("\n" + "★" * 60 + "\n")


if __name__ == "__main__":
    main()
