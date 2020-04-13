# Shivakumar_matrix_operations Package on PyPi

This project was done for basic understanding of uploading a package and gain hands-on experience about object Oriented Programming.

### Summary of The Package
This package containing classes was created as a Project for Udacity Nanodergree progam of Data Scientist. 
Its the 1st project in the Software Engineering section, to create a Package and upload it to PyPi.
This contains simple basic functions in a class to add, subtract, multiply two matrices, obtain transpose of a matrix and multipluing the matrix by a scalar.

### Attributes
Attributes: Class assumes operations will be applied on two distinct matrices. Therefore 2 attributes are created. 
matrix_one: numpy(array) representing first matrix inputed by user. 
matrix_two: numpy(array) representing second matrix inputed by user.


### Methods

####### add_matrices(self): 
Function to add two matrices. 
Args: None Returns: a list or error massage. 
Note: Controls also matrix shapes for math accuracy.

####### subtract_matrices(self): 
Function to subtract two matrices. 
Args: None Returns: a list or error massage. 
Note: Controls also matrix shapes for math accuracy.

###### multiply_matrices(self): 
Function to multiply/dot product two matrices. 
Args: None Returns: a numpy(array) or error massage. 
Note: Controls also matrix shapes for math accuracy.

###### transpose_of_matrix(matrix): 
Method to transpose a matrix. 
Args: given_matrix: numpy(array) or a list 
Returns: a list

###### scalar_multiplication(matrix, value): 
Method to multiply a matrix with a numeric value. 
Args: given_matrix: numpy(array) or a list multiplier: an int or a float 
Returns: a list

### Installation
This package can be instaled from -pip- from pypi. 

Requirements are just numpy. 






