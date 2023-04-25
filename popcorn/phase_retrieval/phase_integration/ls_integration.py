'''
Implementation of the iterative least squares solution

References:
Southwell, J. Opt. Soc. Am, 1980
Li et al, J. Opt. Soc. Am. A, 2013

@Author: Luca Fardin
@Date: 23/02/2023

Main function: least_squares()

Comments: 

          To run sufficiently fast, it needs the scikit-umfpack package. It is not explicitly imported
          but it is called by scipy.sparse.linalg.spsolve

          It is important to define the horizontal direction as x and the vertical direction as y
          Care should be taken in defining the positive spatial direction for the gradient. Currently dx and dy are 
          positive if the column/row index of the matrix is increasing. Different conventions require the multiplication
          of the corresponding Px,Py matrix by -1.
'''

import numpy as np
import time
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


def southwell(gx, gy):
    # This function implements a biquadratic spline fit of the phase
    # The system of equations is given in the form: (P.T P) x = (P.T) S => Ax=b
    # S is a function of the gradients gx,gy. P implements finite differences of the phase.
    # The system of linear equation is solved in the main function by a routine for sparse matrices
    # Input gx: gradient of the phase along the x (h) direction
    #       gy: gradient of the phase along the y (v) direction
    # Output A,b coefficients of the system of linear equations
    #print("southwell sparse")
    N = np.size(gx)
    I, J = np.shape(gx)

    # The matrix P approximates the gradients of the phase "ph" as ph(i+1,j)-ph(i,j) and ph(i,j)-ph(i,j+1)
    # The phase is here a flattened array

    # Initialize the sparse matrix: we know the positions of the non-zero elements
    # We use the lil structure: it is the most efficient for indexing and slicing
    P_lil = lil_matrix((I*(J-1)+J*(I-1),N))

    # Create the non-zero indexes for Px
    rows = np.arange(I*(J-1))
    columns = np.arange(N)
    #We can't use the last element of each row, so we remove them
    columns = np.delete(columns, np.arange(J-1, N, J))
    P_lil[rows, columns] = -1
    P_lil[rows, columns+1] = 1

    # Create the indexes for Py (neighboring points in vertical direction)
    # We can exclude since the beginning the indexes of the last row
    columns=np.arange(J*(I-1))
    rows=columns+I*(J-1)
    P_lil[rows, columns] = -1
    P_lil[rows, columns+J] = 1

    #For matrix multiplication, the most efficient sparse structure is csr
    #We want to solve the system, therfore we use At@A x =At.b
    P = P_lil.tocsr()
    Pt=P.transpose()
    A = Pt @ P

    # The matrix S is the average of contiguous values of gx and gy
    # It is not sparse, therefore we have to use a dense vector
    Sx = 0.5*(gx + np.roll(gx, -1, axis=1))
    Sy = 0.5*(gy + np.roll(gy, -1, axis=0))
    Sxf = Sx[:, :-1].flatten()
    Syf = Sy[:-1, :].flatten()
    S = np.hstack((Sxf, Syf))
    b = Pt.dot(S)

    return A, b


def hfli(gx, gy):
    # This function implements the high-order finite difference based least squares integration.
    # The system of equations is given in the form: (P.T P) x = (P.T) S => Ax=b
    # S is a function of the gradients gx,gy. P implements finite differences of the phase.
    # The system of linear equation is solved in the main function by a routine for sparse matrices.
    # Input gx: gradient of the phase along the x (h) direction
    #       gy: gradient of the phase along the y (v) direction
    # Output A,b coefficients of the system of linear equations
    #print("hfli sparse")
    N = np.size(gx)
    I, J = np.shape(gx)

    # The matrix P contains the zero-th order Taylor expansion of the  phase "ph" as:
    # ph(i+1,j)-ph(i,j) and ph(i,j)-ph(i,j+1)
    # The phase is here a flattened array
    P_lil = lil_matrix((I*(J-1)+J*(I-1),N))

    #For each row, the elements 0,j-1 and j-1 are not considered, because close to the boundary
    rows = np.arange(I*(J-3))
    columns = np.arange(N)
    to_delete = np.concatenate((np.arange(0, N, J), np.arange(J-1, N, J), np.arange(J-2, N, J)))
    columns = np.delete(columns,to_delete)
    P_lil[rows,columns] = -1
    P_lil[rows,columns+1] = 1

    rows = np.arange(I*(J-3),I*(J-3)+J*(I-3))
    columns = np.arange(J,J*(I-2))
    P_lil[rows,columns] = -1
    P_lil[rows,columns+J] = 1

    # We implement additional boundary conditions: Simpson equations
    off = I*(J-3) + J*(I-3)
    for i in np.arange(I):
        P_lil[off+i, i*J] = -1
        P_lil[off+i, i*J+2] = 1
        P_lil[off+I+i, (i+1)*J-3] = -1
        P_lil[off+I+i, (i+1)*J-1] = 1

    #Simpson conditions y direction
    off = I*(J-1) + J*(I-3)
    columns = np.arange(J)
    rows=columns+off
    P_lil[rows,columns] = -1    #Changed 1 to -1
    P_lil[rows,columns+2*J] = 1 #Changed -1 to 1
    P_lil[rows+J,columns+J*(I-3)] = -1   
    P_lil[rows+J,columns+J*(I-1)] = 1  
    # S is the matrix containing the higher order terms of the Taylor expansion

    P = P_lil.tocsr()
    Pt=P.transpose()
    A = Pt @ P

    #print("Generating gradient matrix")
    Sx = 13/24.*(gx - 1/13.*np.roll(gx, 1, axis=1) +
                 np.roll(gx, -1, axis=1) - 1/13. * np.roll(gx, -2, axis=1))
    Sx = np.delete(Sx, (0, J-2, J-1), axis=1)
    Sy = 13/24.*(gy - 1/13.*np.roll(gy, 1, axis=0) +
                 np.roll(gy, -1, axis=0)-1/13.*np.roll(gy, -2, axis=0))
    Sy = np.delete(Sy, (0, I-2, I-1), axis=0)
    Sxf = Sx.flatten()
    Syf = Sy.flatten()
    S = np.hstack((Sxf, Syf))

    S_Sx = 1/3 * (gx + 4*np.roll(gx, -1, axis=1) + np.roll(gx, -2, axis=1))
    S_Sy = 1/3 * (gy + 4*np.roll(gy, -1, axis=0) + np.roll(gy, -2, axis=0))
    S_Sxf = S_Sx[:, (0, J-3)].flatten('F')
    S_Syf = S_Sy[(0, I-3), :].flatten()
    S = np.hstack((S, S_Sxf, S_Syf))
    b = Pt.dot(S)

    return A, b


def least_squares(gx, gy, px, py, model='southwell'):

    # This is the main function to determine the phase from iterative least squares solution
    # Input:  gx, gy : gradient along x (h) and y (v) respectively
    #         px, py : pixel size
    #         model  : least squares implementation 'southwell', 'hfli'
    # Output: phase  :Reconstructed phase image

    print("Least Square Solver: {}".format(model))

    # 1st step: normalize the vectors to work in pixel units
    gxn = gx*px
    gyn = gy*py

    # Least squares is based on a system of the form Ax=b.
    # A and b depend on the order of the finite differences chosen.

    start = time.time()
    model_func = globals()[model]
    A, b = model_func(gxn, gyn)

    #Solve the system using the fast umfpack solver for sparse matrices
    phase = spsolve(A, b,use_umfpack=True)
    stop = time.time()
    print("execution time = :", stop-start)
    phase = np.reshape(phase, np.shape(gx))
    return phase
