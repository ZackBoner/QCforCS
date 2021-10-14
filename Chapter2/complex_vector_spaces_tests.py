from complex_vector_spaces import ComplexVector, dot, nvector_inner_product, distance, tensor_product


if __name__ == "__main__":
    A = ComplexVector([[1, 2], [0, 1]])
    B = ComplexVector([[3, 2], [-1, 0]])
    C = ComplexVector([[6, 5], [3, 2]])

    print(tensor_product(A, tensor_product(B, C)))
    print()
    print(tensor_product(tensor_product(A, B), C))
