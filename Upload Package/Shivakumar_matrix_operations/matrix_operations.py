import numpy as np

class MatrixOperations:
    """
    This class is created for Udacity Data Scientist Nanodegree program : 
    Upload a package to PyPi.
    
    Matrix operations are done for addition, subtraction, unequal addition, 
    matrix multiplication, transpose calculation and scalar multiplication.
    
    Attributes:
    Operations performed on 2 Matrices. 
    matrix_one = numpy array representing first matrix inputed by user.
    matrix_two = numpy array representing seconf matrix inputed by user.
"""


    def __init__(self, matrix_one, matrix_two):
        self.matrix_one = matrix_one
        self.matrix_two = matrix_two
        
    def add_matrices(self):
        """
        Args: None
        Returns: a list or Error
        Note: Shape of matrix also considered
        """
        result = np.zeros((self.matrix_one.shape[0], self.matrix_one.shape[1]))
        if self.matrix_one.shape == self.matrix_two.shape:
            for i in range(len(self.matrix_one)):
                for j in range(len(self.matrix_one[0])):
                    result[i,j] = self.matrix_one[i,j] + self.matrix_two[i,j]
            return result
        else:
            print("Math error")
            print("Addition cannot be performed, Shape of the matrices must be same.")
            
    def subtract_matrices(self):
        result = np.zeros((self.matrix_one.shape[0], self.matrix_one.shape[1]))
        if self.matrix_one.shape == self.matrix_two.shape:
            for i in range(len(self.matrix_one)):
                for j in range(len(self.matrix_one[0])):
                    result[i,j] = self.matrix_one[i,j] - self.matrix_two[i,j]
            return result
        else:
            print("Math error")
            print("Subtraction cannot be performed, Shape of the matrices must be same.")
            
    def multiply_matrices(self):
        """
        Args: None
        Returns : A numpy array or Error message.
        Notes: Depends on shape of matrix.
        
        """
        
        final_matrix = np.zeros((self.matrix_one.shape[0], self.matrix_two.shape[1]))
        if self.matrix_one.shape[1] == self.matrix_two.shape[0]:
            for i in range(self.matrix_one.shape[0]):
                for j in range(self.matrix_two.shape[0]):
                    final_matrix[i] += self.matrix_one[i,j] * self.matrix_two[j,i]
            return final_matrix
        else:
            print("Math Error")
            print("Multiplication cannot be performeeed. No. of columns on 1st matrix should be equal to No. of rows on 2nd matrix")
            
    @staticmethod
    def transpose_of_matrix(matrix):
        final_matrix = np.zeros((matrix.shape[1], matrix.shape[0]))

        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[0]):
                final_matrix[i,j] =  matrix[j,i]
        return final_matrix
                
    @staticmethod        
    def scalar_multiplication(matrix, value):
        final_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                final_matrix[i,j] = value * matrix[i,j]
        return final_matrix


    