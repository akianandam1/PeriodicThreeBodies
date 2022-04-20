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
input_vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0, 35, 35, 35], requires_grad = True).to(device)
input_vec.retain_grad()

def forward(input_vec):
    return get_full_state(input_vec, 0.001, 10)


# Compares two states and returns a numerical value rating how far aart the two states in the three
# body problem are to each other. Takes in two tensor states and returns tensor of value of distance rating
# The higher the score, the less similar they are
def compare_states(state1, state2):
    mse = torch.nn.MSELoss()
    return mse(state1, state2)

def most_similar_state(data_set):
    i = 3
    max_val = 100000000
    index = -1
    while i < len(data_set):
        if 3*torch.dot(data_set[0][:9], data_set[i][:9]).item() + torch.dot(data_set[0][9:18], data_set[i][9:18]).item() < max_val:
            index = i
            max_val = 3*torch.dot(data_set[0][:9], data_set[i][:9]).item() + torch.dot(data_set[0][9:18], data_set[i][9:18]).item()

        i += 1
    print(f"Idex: {index}")
    return index


def fit(epochs, lr, input_vec):
    i = 0
    while i < epochs:

        data_set = forward(input_vec)

        state = data_set[most_similar_state(data_set)][:18]

        gain = 3*torch.dot(data_set[0][:9], state[:9]) + torch.dot(data_set[0][9:18], state[9:18])
        print(" ")
        print(gain)

        #loss.retain_grad()
        gain.backward()
        print(input_vec.grad)
        # Updates input vector

        input_vec += input_vec.grad * lr
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
a=100
b=.0001
fit(a, b, input_vec)
with open("mainoutput.txt", "a") as file:
    file.write("\n")
    file.write(f"{a}, {b}: \n")
    file.write(str(input_vec))
print(input_vec)

