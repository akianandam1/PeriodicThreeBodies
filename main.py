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


def forward(input_vec):
    return get_full_state(input_vec, 0.001, 10)


# Compares two states and returns a numerical value rating how far apart the two states in the three
# body problem are to each other. Takes in two tensor states and returns tensor of value of distance rating
# The higher the score, the less similar they are
def compare_states(state1, state2):
    mse = torch.nn.MSELoss()
    return 3*mse(state1[:9], state2[:9]) + mse(state1[9:18], state2[9:18])


# Finds the most similar state to the initial position in a data set
def most_similar_state(data_set, min, max):
    i = min
    max_val = torch.tensor([100000000]).to(device)
    index = -1
    while i < max:
        if compare_states(data_set[0], data_set[i]) < max_val:
            index = i
            max_val = compare_states(data_set[0], data_set[i])

        i += 1
    print(f"Index: {index}")
    return index


def fit(epochs, lr, input_vec):
    i = 0
    while i < epochs:

        data_set = forward(input_vec)

        first_state_index = most_similar_state(data_set, 300, len(data_set))
        first_state = data_set[first_state_index]
        first_state.retain_grad()
        second_state_index = most_similar_state(data_set, int(1.8*first_state_index), int(2.2*first_state_index))
        second_state = data_set[second_state_index]
        second_state.retain_grad()
        third_state_index = most_similar_state(data_set, int(2.6*first_state_index), int(3.4*first_state_index))
        third_state = data_set[third_state_index]
        third_state.retain_grad()

        loss = compare_states(data_set[0], first_state) + compare_states(data_set[0], second_state) + compare_states(data_set[0], third_state)
        loss.retain_grad()
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

