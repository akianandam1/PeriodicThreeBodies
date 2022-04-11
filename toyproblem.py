import torch
from TorchNumericalSolver import get_final_data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn

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


# Input vector we will be optimizing which represents the velocity
# components of our malleable particle.
input_vec = torch.tensor([0.5,-0.5,0.3], dtype = torch.float32, requires_grad=True)

def forward(input_vec):
    # p1 starts at (1,0,0)
    # p2 starts at (0,0,0)
    # p3 starts at (0,1,0)
    # p1 is the velocity we are changing. p2 and p3 have zero velocity
    # all have mass 1

    full_vec = torch.cat([torch.tensor([1,0,0,0,0,0,0,1,0]),input_vec, torch.tensor([0,0,0,0,0,0,1,1,1])])

    # returns position of p1 after 3 time units
    return get_final_data(full_vec)[0:3]


def get_loss(position, desired_position):
    return torch.sqrt((desired_position[0]-position[0])**2 + (desired_position[1]-position[1])**2 + (desired_position[2]-position[2])**2)


# Defines the function that trains the model
def fit(epochs, lr, input_vec):

    # Gets appropriate batch size such that we use all the data by the time we finish our epochs
    # optimizer = opt_func(input, lr=lr)

    i = 0
    while i < epochs:

        # p1 should end up at (2,0,0)
        loss = get_loss(forward(input_vec), torch.tensor([2, 0, 0]))

        loss.backward()

        # Updates input vector
        with torch.no_grad():
            input_vec -= input_vec.grad*lr

        # Zeroes gradient
        input_vec.grad.zero_()

        # optimizer.step()
        # optimizer.zero_grad()
        print(i)
        i += 1


print(input_vec)
fit(1000, 0.001, input_vec)
print(input_vec)