from TorchNumericalSolver import get_full_state
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import torch


mpl.rcParams['animation.ffmpeg_path'] = r'D:\Aki\Python37\Lib\site-packages\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'


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



fig = plt.figure(figsize=(20, 20))
ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3))
ax.set_title("Visualization of orbits of stars in a three body system\n", fontsize=28)
particle1, = plt.plot([], [], color='r', label="Real First Star")
particle2, = plt.plot([], [], color='g', label="Real Second Star")
particle3, = plt.plot([], [], color='b', label="Real Third Star")
ax.legend(loc="upper left", fontsize=28)


# Starting initial vector
vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0, 35.7071, 35.7071, 35.7071], requires_grad = True).to(device)


def init():
    particle1, = plt.plot([], [], color='r', label="Real First Star")
    particle2, = plt.plot([], [], color='g', label="Real Second Star")
    particle3, = plt.plot([], [], color='b', label="Real Third Star")
    return particle1, particle2, particle3


def update(i, lr, input_vec):
    data_set = forward(input_vec)
    first_particle_state = data_set[nearest_position_state(1, data_set[0], data_set, 300, len(data_set))]
    second_particle_state = data_set[nearest_position_state(2, data_set[0], data_set, 300, len(data_set))]
    third_particle_state = data_set[nearest_position_state(3, data_set[0], data_set, 300, len(data_set))]
    loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0],
                                                                                     second_particle_state) + nearest_position(
        3, data_set[0], third_particle_state)

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

    # with open("tempvec.py", "w") as file:
    #     stringvec = str(vec)[:str(vec).index("]")+1]
    #     file.truncate(0)
    #     file.write(f"import torch\ninput_vec = torch.{stringvec}, requires_grad=True)")

    data_set = data_set.cpu().detach().numpy()
    print(f"data_set: {data_set[:,0]}")

    particle1, = plt.plot(data_set[:, 0], data_set[:, 1], color='r', label="Real First Star")
    particle2, = plt.plot(data_set[:, 3], data_set[:, 4], color='g', label="Real Second Star")
    particle3, = plt.plot(data_set[:, 5], data_set[:, 6], color='b', label="Real Third Star")

    print(f"Epoch:{i}")
    print(" ")
    return particle1, particle2, particle3


writer = animation.FFMpegWriter(fps=50)
ani = animation.FuncAnimation(fig, update, frames=1000, fargs=(.00001, vec))
ani.save(r"D:\Aki\Pycharm\PycharmProjects\PeriodicThreeBodies\Videos\May22\a5.mp4", writer=writer)

a = 1000
b = .00001

with open("mainoutput.txt", "a") as file:
    file.write("\n")
    file.write(f"{a}, {b}: \n")
    file.write(str(vec))

print(vec)

# reset_input = input("Press enter to reset input:")
# with open("tempvec.py", "w") as file:
#     file.truncate(0)
#     file.write("import torch\ninput_vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0, 35.7071, 35.7071, 35.7071], requires_grad = True)")
#
