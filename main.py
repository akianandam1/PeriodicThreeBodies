import torch
from TorchNumericalSolver import get_full_state


# Function to determine whether gpu is available or not
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


# Automatically puts data onto the default device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Starting initial vector
#input_vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0, 35, 35, 35], requires_grad = True).to(device)
input_vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0, 35.7071, 35.7071, 35.7071], requires_grad = True).to(device)

def forward(input_vec):
    return get_full_state(input_vec, 0.001, 5)


# Compares two states and returns a numerical value rating how far apart the two states in the three
# body problem are to each other. Takes in two tensor states and returns tensor of value of distance rating
# The higher the score, the less similar they are
def nearest_position(particle, state1, state2):
    mse = torch.nn.L1Loss()
    if particle == 1:
        return mse(state1[:3], state2[:3]) + mse(state1[9:12], state2[9:12])
    elif particle == 2:
        return mse(state1[3:6], state2[3:6]) + mse(state1[12:15], state2[12:15])
    elif particle == 3:
        return mse(state1[6:9], state2[6:9]) + mse(state1[15:18], state2[15:18])
    else:
        print("bad input")


# Finds the most similar state to the initial position in a data set
def nearest_position_state(particle, state, data_set, min, max):
    i = min
    max_val = torch.tensor([100000000]).to(device)
    index = -1
    while i < max:
        if nearest_position(particle, state, data_set[i]).item() < max_val.item():
            index = i
            max_val = nearest_position(particle, state, data_set[i])

        i += 1
    print(f"Index: {index}")
    return index



# def nearest_two_states(data_set, min):
#     i = 0
#     first_index = -1
#     second_index=-1
#     max_val = torch.tensor([100000000]).to(device)
#     while i < (len(data_set)/2 + 1):
#         print(i)
#         if nearest_position(data_set[i], data_set[nearest_position_state(data_set[i], data_set, i + min, len(data_set))]) < max_val:
#             first_index = i
#             second_index = nearest_position_state(data_set[i], data_set, i + min, len(data_set))
#             max_val = nearest_position(data_set[i], data_set[nearest_position_state(data_set[i], data_set, i + min, len(data_set))])
#         i+=1
#     print(f"Index: {first_index}, {second_index}")
#     return torch.tensor([first_index,second_index]).to(device)
#
#
# data_set = forward(input_vec)
#
#
# print(nearest_two_states(data_set, 500))


def compare_states(state1, state2):
    mse = torch.nn.MSELoss()
    return mse(state1[:18], state2[:18])


def fit(epochs, lr, input_vec):
    i = 0
    while i < epochs:

        data_set = forward(input_vec)
        first_particle_state = data_set[nearest_position_state(1, data_set[0],data_set, 300, len(data_set))]
        second_particle_state = data_set[nearest_position_state(2, data_set[0], data_set, 300, len(data_set))]
        third_particle_state = data_set[nearest_position_state(3, data_set[0], data_set, 300, len(data_set))]

        #first_state_index = nearest_position_state(data_set[0],data_set, 300, len(data_set))
        # first_state = data_set[first_state_index]
        # first_state.retain_grad()
        # second_state_index = most_similar_state(data_set, int(1.8*first_state_index), int(2.2*first_state_index))
        # second_state = data_set[second_state_index]
        # second_state.retain_grad()
        # third_state_index = most_similar_state(data_set, int(2.6*first_state_index), int(3.4*first_state_index))
        # third_state = data_set[third_state_index]
        # third_state.retain_grad()

        loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0], second_particle_state) + nearest_position(3, data_set[0], third_particle_state)
        print(" ")
        print(loss)
        input_vec.retain_grad()
        #loss.retain_grad()
        loss.backward()
        print(input_vec.grad)
        # Updates input vector

        input_vec -= input_vec.grad * lr
        print(input_vec)
        #print(input_vec.grad)
        # Zeroes gradient
        input_vec.grad.zero_()

        # optimizer.step()
        # optimizer.zero_grad()
        print(f"Epoch:{i}")
        print(" ")
        i += 1


print(input_vec)
a=1000
b=.00001
fit(a, b, input_vec)
with open("mainoutput.txt", "a") as file:
    file.write("\n")
    file.write(f"{a}, {b}: \n")
    file.write(str(input_vec))
print(input_vec)

