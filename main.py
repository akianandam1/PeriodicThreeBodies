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
input_vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0, 35, 35, 35], requires_grad = True)

def forward(input_vec):
    return get_full_state(input_vec, 0.001, 10)

def most_similar_state(input_vec, data_set):

    with torch.no_grad():
        i = 0
        max_val = -100000000
        index = -1
        while i < len(data_set):
            if torch.dot(input_vec, data_set[i]).item() > max_val:
                index = i

            i += 1
        return index


def fit(epochs, lr, input_vec):
    i = 0
    while i < epochs:
        data_set = forward(input_vec)
        state = data_set[most_similar_state(input_vec, data_set)]
        loss = -torch.dot(input_vec, state)
        #loss.retain_grad()
        loss.backward()

        # Updates input vector
        with torch.no_grad():
            input_vec -= input_vec.grad * lr

        # Zeroes gradient
        input_vec.grad.zero_()

        # optimizer.step()
        # optimizer.zero_grad()
        print(i)
        i += 1


print(input_vec)
fit(10, 0.00001, input_vec)
print(input_vec)

