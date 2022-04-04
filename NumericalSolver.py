import numpy as np
import scipy as sci
from scipy import linalg
from scipy.integrate import odeint
from constants import *

# Differential Equations Governing Three Bodies
# w is flattened input array with position vector followed by velocity vector
def ThreeBodyDiffEq(w,t, m_1, m_2, m_3):

    # Unpacks flattened array
    r_1 = w[:3]
    r_2 = w[3:6]
    r_3 = w[6:9]
    v_1 = w[9:12]
    v_2 = w[12:15]
    v_3 = w[15:18]

    # Displacement vectors
    r_12 = linalg.norm(r_2 - r_1)
    r_13 = linalg.norm(r_3 - r_1)
    r_23 = linalg.norm(r_3 - r_2)

    # The derivatives of the velocities
    dv_1bydt = K_1 * m_2 * (r_2 - r_1) / r_12 ** 3 + K_1 * m_3 * (r_3 - r_1) / r_13 ** 3
    dv_2bydt = K_1 * m_1 * (r_1 - r_2) / r_12 ** 3 + K_1 * m_3 * (r_3 - r_2) / r_23 ** 3
    dv_3bydt = K_1 * m_1 * (r_1 - r_3) / r_13 ** 3 + K_1 * m_2 * (r_2 - r_3) / r_23 ** 3

    # The derivatives of the positions
    dr_1bydt = K_2 * v_1
    dr_2bydt = K_2 * v_2
    dr_3bydt = K_2 * v_3

    # Vector in form [position derivatives, velocity derivatives]
    derivatives = np.array([[dr_1bydt, dr_2bydt, dr_3bydt], [dv_1bydt, dv_2bydt, dv_3bydt]])

    # In order for Scipy module to use this it must be a 1d array
    return derivatives.flatten()



# Input vector is np array and has form [time,
#                                        particle 1 position vector,
#                                        particle 2 position vector,
#                                        particle 3 position vector,
#                                        particle 1 velocity vector,
#                                        particle 2 velocity vector,
#                                        particle 3 velocity vector,
#                                        particle 1 mass,
#                                        particle 2 mass,
#                                        particle 3 mass]
# for a total of 22 parameters. AlL the vectors are flattened. No arrays within arrays
def numerical_solver(input_vector):

    # Gets the components
    time = input_vector[0]
    r_1 = input_vector[1:4]
    r_2 = input_vector[4:7]
    r_3 = input_vector[7:10]
    v_1 = input_vector[10:13]
    v_2 = input_vector[13:16]
    v_3 = input_vector[16:19]
    m_1 = input_vector[19]
    m_2 = input_vector[20]
    m_3 = input_vector[21]

    # vector comprised of position and velocity
    w = np.array([[r_1,r_2,r_3], [v_1,v_2,v_3]])
    # flattens for scipy to use
    w = w.flatten()

    # Time points for the numerical diff eq solver to use. Spans from 0 to t
    # and has points that are .001 time units spaced apart
    time_points = np.linspace(0, time, int(time/.001))

    # input into scipy odeint must be 1 dimensional
    # Gets the solutions to the differential equations
    three_body_solution = sci.integrate.odeint(ThreeBodyDiffEq, w, time_points, args = (m_1, m_2, m_3))

    # Extracts the position aspect of the solutions
    r_1_solution = three_body_solution[:, :3]
    r_2_solution = three_body_solution[:, 3:6]
    r_3_solution = three_body_solution[:, 6:9]
    # Builds output vector where we get the last value of each position (corresponding
    # to where the particle is at when time equals t, then returns array
    # comprised of these vectors
    output = r_1_solution[-1]
    output = np.append(output, r_2_solution[-1])
    output = np.append(output, r_3_solution[-1])
    return output



def get_full_data(input_vector):

    # Gets the components
    time = 5 # Assumes time component is 10
    r_1 = input_vector[0:3]
    r_2 = input_vector[3:6]
    r_3 = input_vector[6:9]
    v_1 = input_vector[9:12]
    v_2 = input_vector[12:15]
    v_3 = input_vector[15:18]
    m_1 = input_vector[18]
    m_2 = input_vector[19]
    m_3 = input_vector[20]

    # vector comprised of position and velocity
    w = np.array([[r_1,r_2,r_3], [v_1,v_2,v_3]])
    # flattens for scipy to use
    w = w.flatten()

    # Time points for the numerical diff eq solver to use. Spans from 0 to t
    # and has points that are .01 time units spaced apart
    time_points = np.linspace(0, time, int(time/.001))

    # input into scipy odeint must be 1 dimensional
    # Gets the solutions to the differential equations
    three_body_solution = sci.integrate.odeint(ThreeBodyDiffEq, w, time_points, args = (m_1, m_2, m_3))

    output = three_body_solution[:, 0:9]
    return output.flatten()

