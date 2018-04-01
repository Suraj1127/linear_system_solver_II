#!/usr/bin/python3
import numpy as np

class LinearSystem:
    def __init__(self, no_of_equations, no_of_variables, A, b):
        """
        :param no_of_equations: number of equations of the linear system
        :param no_of_variables: number of variables of the linear system
        :A: augmented matrix A containing coefficients of the linear system and b
        """
        self.no_of_equations = no_of_equations
        self.no_of_variables = no_of_variables
        self.A = np.append(A,b,axis=1)

    def row_echelonize(self):
        """
        reduce the matrix augmented matrix A into echelon form
        """
        last_pivot_row = -1
        for i in range(self.no_of_variables):
            if (self.A[last_pivot_row+1:,i] == np.zeros(self.no_of_equations-last_pivot_row-1)).all():
                continue
            pivot = self.A[last_pivot_row+1][i]
            if pivot == 0:
                for j in range(last_pivot_row+2,self.no_of_equations):
                    if self.A[j][i] != 0:
                        temp1 = np.copy(self.A[last_pivot_row+1])
                        temp2 = np.copy(self.A[j])
                        self.A[j], self.A[last_pivot_row+1] = temp1, temp2
                        break

            pivot = self.A[last_pivot_row+1][i]
            self.A[last_pivot_row+1] = self.A[last_pivot_row+1]/pivot
            last_pivot_row += 1

            for j in range(last_pivot_row+1, self.no_of_equations):
                if self.A[j][i] != 0:
                    self.A[j] = self.A[j] - self.A[j][i]*self.A[last_pivot_row]

    def reduced_row_echelonize(self):
        """
        reduce the row echelon form of A into reduced row echelon form
        """
        rank = 1
        pivot_columns = [0]
        for i in range(1, self.no_of_equations):
            try:
                pivot_column = list(self.A[i][0:self.no_of_variables]).index(1)
            except ValueError:
                if self.A[i][-1] != 0:
                    print("\nThe linear system has no solution.")
                    exit()
                else:
                    continue
                break
            rank += 1
            pivot_columns.append(pivot_column)
            for j in range(0, i):
                self.A[j] = self.A[j]-self.A[j,pivot_column]*self.A[i]
        self.rank = rank
        self.pivot_columns = pivot_columns
        self.free_columns = sorted(list(set(range(self.no_of_variables)) - set(pivot_columns)))

    def get_free_matrix(self):
        """
        returns the free matrix consisting of free variable columns of A leaving b
        """
        self.free_matrix = np.zeros((self.rank,self.no_of_variables-self.rank))
        for i in range(self.free_matrix.shape[1]):
            self.free_matrix[:,i] = self.A[:self.rank,self.free_columns[i]]
        return self.free_matrix

    def get_null_space(self):
        """
        returns the null space of the matrix A leaving b column
        """
        self.get_free_matrix()
        if self.free_matrix.shape[1] == 0:
            return np.zeros((self.no_of_variables,1))
        self.null_space = np.zeros((self.free_matrix.shape[0]+self.free_matrix.shape[1],len(self.free_columns)))
        identity_matrix = np.eye(len(self.free_columns))
        j = 0
        for i in self.pivot_columns:
            self.null_space[i] = -self.free_matrix[j]
            j += 1
        j = 0
        for i in self.free_columns:
            self.null_space[i] = identity_matrix[j]
            j += 1
        return self.null_space

    def get_particular_solution(self):
        """
        returns particular solution for the system given it in reduced
        row echelon form
        """
        self.particular_solution = np.zeros((self.no_of_variables,1))
        j = 0
        for i in self.pivot_columns:
            self.particular_solution[i][0] = self.A[:,-1][j]
            j += 1


def main():
    # input for number of variables and equations
    print("Enter the number of variables:")
    no_of_variables = int(input())
    print("Enter the number of equations:")
    no_of_equations = int(input())

    # input for matrix A and setting to linear system
    A = np.zeros((no_of_equations, no_of_variables))
    print("Enter the coefficients of equations line by line separated by a single space:")
    print("\"\"\"\nFor example for 3 equations and 4 variables, the coefficient matrix A would be of the form:")
    print("4 4 2 3\n6 4 5 3\n3 6 4 2\n\"\"\"")
    for i in range(no_of_equations):
        A[i] = [int(number) for number in input().split()]

    # input for vector b
    print("\nEnter the RHS vector b with components separated by spaces:")
    print("\"\"\"For 4 dimensional vector, example would be: 4 2 4 3\"\"\"")
    b = np.array([int(i) for i in input().split()]).reshape(no_of_equations,1)

    linear_system = LinearSystem(no_of_equations, no_of_variables, A, b)
    linear_system.row_echelonize()
    linear_system.reduced_row_echelonize()
    linear_system.get_particular_solution()

    print("\nThe solution for this linear system is:\n")
    print("Column space of:\n{}\n+\n{}".format(linear_system.get_null_space(), linear_system.particular_solution))


if __name__ == "__main__":
    main()


