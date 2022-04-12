import torch
from constants import *
import time

# Differential Equations Governing Three Bodies
# w is flattened input torch tensor with position vector followed by velocity vector
def ThreeBodyDiffEq(w, m_1, m_2, m_3):
    # Unpacks flattened array
    r_1 = w[:3]
    r_2 = w[3:6]
    r_3 = w[6:9]
    v_1 = w[9:12]
    v_2 = w[12:15]
    v_3 = w[15:18]

    # Torch calculated displacement vector magnitudes
    r_12 = torch.sqrt((r_2[0] - r_1[0]) ** 2 + (r_2[1] - r_1[1]) ** 2 + (r_2[2] - r_1[2]) ** 2)
    r_13 = torch.sqrt((r_3[0] - r_1[0]) ** 2 + (r_3[1] - r_1[1]) ** 2 + (r_3[2] - r_1[2]) ** 2)
    r_23 = torch.sqrt((r_3[0] - r_2[0]) ** 2 + (r_3[1] - r_2[1]) ** 2 + (r_3[2] - r_2[2]) ** 2)

    # The derivatives of the velocities. Returns torch tensor
    dv_1bydt = K_1 * m_2 * (r_2 - r_1) / r_12 ** 3 + K_1 * m_3 * (r_3 - r_1) / r_13 ** 3
    dv_2bydt = K_1 * m_1 * (r_1 - r_2) / r_12 ** 3 + K_1 * m_3 * (r_3 - r_2) / r_23 ** 3
    dv_3bydt = K_1 * m_1 * (r_1 - r_3) / r_13 ** 3 + K_1 * m_2 * (r_2 - r_3) / r_23 ** 3

    # The derivatives of the positions
    dr_1bydt = K_2 * v_1
    dr_2bydt = K_2 * v_2
    dr_3bydt = K_2 * v_3

    # Vector in form [position derivatives, velocity derivatives]
    derivatives = torch.stack([dr_1bydt, dr_2bydt, dr_3bydt, dv_1bydt, dv_2bydt, dv_3bydt])

    # In order for Scipy module to use this it must be a 1d array
    return derivatives.flatten()


# Returns torch tensor of the solutions to the diff eq satisfied by input vector, w/ small step dt and
# total time time_span. Takes in flat 21 dimension input torch tensor
def torchIntegrate(input_vector, dt, time_span):
    # Gets values from input vector. These will be torch tensors
    r_1 = input_vector[0:3]
    r_2 = input_vector[3:6]
    r_3 = input_vector[6:9]
    v_1 = input_vector[9:12]
    v_2 = input_vector[12:15]
    v_3 = input_vector[15:18]
    m_1 = input_vector[18].item()
    m_2 = input_vector[19].item()
    m_3 = input_vector[20].item()

    w = torch.stack(
        [r_1.flatten(), r_2.flatten(), r_3.flatten(), v_1.flatten(), v_2.flatten(), v_3.flatten()]).flatten()

    # Number of times we iterate over
    number_of_points = time_span / dt
    i = 1
    results = torch.stack([r_1, r_2, r_3]).flatten()[None, :]

    while (i < number_of_points):
        derivatives = ThreeBodyDiffEq(w, m_1, m_2, m_3)

        # Updates values of position and velocity vectors after step
        r_1 = torch.stack([r_1[0] + derivatives[0] * dt, r_1[1] + derivatives[1] * dt,
                           r_1[2] + derivatives[2] * dt])
        r_2 = torch.stack([r_2[0] + derivatives[3] * dt, r_2[1] + derivatives[4] * dt,
                           r_2[2] + derivatives[5] * dt])
        r_3 = torch.stack([r_3[0] + derivatives[6] * dt, r_3[1] + derivatives[7] * dt,
                           r_3[2] + derivatives[8] * dt])
        v_1 = torch.stack([v_1[0] + derivatives[9] * dt, v_1[1] + derivatives[10] * dt,
                           v_1[2] + derivatives[11] * dt])
        v_2 = torch.stack([v_2[0] + derivatives[12] * dt, v_2[1] + derivatives[13] * dt,
                           v_2[2] + derivatives[14] * dt])
        v_3 = torch.stack([v_3[0] + derivatives[15] * dt, v_3[1] + derivatives[16] * dt,
                           v_3[2] + derivatives[17] * dt])

        # Appends new positions to results
        results = torch.cat((results, torch.stack([r_1, r_2, r_3]).flatten()[None, :]))
        w = torch.stack(
            [r_1.flatten(), r_2.flatten(), r_3.flatten(), v_1.flatten(), v_2.flatten(), v_3.flatten()]).flatten()
        print(i)
        i += 1

    return results

def get_full_state(input_vector, dt, time_span):
    r_1 = input_vector[0:3]
    r_2 = input_vector[3:6]
    r_3 = input_vector[6:9]
    v_1 = input_vector[9:12]
    v_2 = input_vector[12:15]
    v_3 = input_vector[15:18]
    m_1 = input_vector[18].item()
    m_2 = input_vector[19].item()
    m_3 = input_vector[20].item()

    w = torch.stack(
        [r_1.flatten(), r_2.flatten(), r_3.flatten(), v_1.flatten(), v_2.flatten(), v_3.flatten()]).flatten()

    # Number of times we iterate over
    number_of_points = time_span / dt
    i = 1
    results = torch.stack([r_1, r_2, r_3, v_1, v_2, v_3]).flatten()[None, :]

    results = torch.cat((results.flatten(), torch.stack([input_vector[18], input_vector[19], input_vector[20]]).flatten()))[None, :]

    while (i < number_of_points):
        derivatives = ThreeBodyDiffEq(w, m_1, m_2, m_3)

        # Updates values of position and velocity vectors after step
        r_1 = torch.stack([r_1[0] + derivatives[0] * dt, r_1[1] + derivatives[1] * dt,
                           r_1[2] + derivatives[2] * dt])
        r_2 = torch.stack([r_2[0] + derivatives[3] * dt, r_2[1] + derivatives[4] * dt,
                           r_2[2] + derivatives[5] * dt])
        r_3 = torch.stack([r_3[0] + derivatives[6] * dt, r_3[1] + derivatives[7] * dt,
                           r_3[2] + derivatives[8] * dt])
        v_1 = torch.stack([v_1[0] + derivatives[9] * dt, v_1[1] + derivatives[10] * dt,
                           v_1[2] + derivatives[11] * dt])
        v_2 = torch.stack([v_2[0] + derivatives[12] * dt, v_2[1] + derivatives[13] * dt,
                           v_2[2] + derivatives[14] * dt])
        v_3 = torch.stack([v_3[0] + derivatives[15] * dt, v_3[1] + derivatives[16] * dt,
                           v_3[2] + derivatives[17] * dt])

        # Appends new positions to results
        new_results = torch.cat((torch.stack([r_1, r_2, r_3, v_1, v_2, v_3, ]).flatten(), torch.stack([input_vector[18], input_vector[19], input_vector[20]]).flatten()))[None, :]
        results = torch.cat((results, new_results))

        w = torch.stack(
            [r_1.flatten(), r_2.flatten(), r_3.flatten(), v_1.flatten(), v_2.flatten(), v_3.flatten()]).flatten()
        #print(i)
        i += 1

    return results




# Function for getting the final position of objects after time elapse w/ dt time interval
def get_final_data(input_vector, dt=.001, time_span=1):
    return torchIntegrate(input_vector, dt, time_span)[-1]


# .001 dt for 100 s
if __name__ == "__main__":

    input_vec = torch.tensor([2.3050e-01, -4.0595e-02, 8.5479e-44])
    full_vec = torch.cat(
        [torch.tensor([1, 0, 0, 0, 0, 0, 0, 1, 0]), input_vec, torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])])

    sol = get_full_state(full_vec, .01, 10)
    print(sol.shape)
    # ans = torchIntegrate(full_vec, .001, 10).numpy()
    # np.set_printoptions(threshold=sys.maxsize)
    # print(ans)
