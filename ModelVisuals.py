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
    return get_full_state(input_vec, 0.0001, 5)


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


figure, ax = plt.subplots(2, 1)
top, bottom = ax
top.set_xlim(-3,3)
top.set_ylim(-3,3)
# fig = plt.figure(figsize=(20, 20))
# ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3))
# ax.set_title("Visualization of orbits of stars in a three body system\n", fontsize=28)
particle1, = top.plot([], [], color='r', label="Real First Star")
particle2, = top.plot([], [], color='g', label="Real Second Star")
particle3, = top.plot([], [], color='b', label="Real Third Star")
first_text = bottom.text(0.7, 0.85, "", fontsize = "xx-small", transform=ax[1].transAxes)
second_text = bottom.text(0.7, 0.78, "", fontsize = "xx-small", transform = ax[1].transAxes)
third_text = bottom.text(0.7, 0.71, "", fontsize = "xx-small", transform = ax[1].transAxes)

# ax.legend(loc="upper left", fontsize=28)
top.legend(loc="upper left", fontsize=6)

# Starting initial vector
#vec = torch.tensor([-1, 0, 0, 1, 0, 0, 0, 0, 0, 0.347111, 0.532728, 0, 0.347111, 0.532728, 0, -2*0.347111, -2*0.532728, 0, 35.7071, 35.7071, 35.7071], requires_grad = True).to(device)

vec = torch.tensor([ 1.5204e+00,  7.6287e-01, -7.2413e-02, -1.7299e-01,  4.2930e-01,
        -9.4932e-02, -6.4748e-01, -3.9216e-01, -3.2655e-02,  7.0786e-01,
         5.1624e-01, -1.6740e-04,  2.0788e-01,  3.1259e-01, -2.9388e-02,
        -8.1860e-01, -7.8648e-01,  2.8100e-02,  3.5703e+01,  3.5734e+01,
         3.5692e+01], requires_grad=True).to(device)



def init():
    particle1, = top.plot([], [], color='r', label="Real First Star")
    particle2, = top.plot([], [], color='g', label="Real Second Star")
    particle3, = top.plot([], [], color='b', label="Real Third Star")
    first_text = bottom.text(0.7, 0.85, "", fontsize = "xx-small", transform=ax[1].transAxes)
    second_text = bottom.text(0.7, 0.78, "", fontsize = "xx-small", transform = ax[1].transAxes)
    third_text = bottom.text(0.7, 0.71, "", fontsize = "xx-small", transform = ax[1].transAxes)

    return particle1, particle2, particle3, first_text, second_text, third_text

loss_values = []


def update(i, lr, input_vec):
    data_set = forward(input_vec)
    first_index = nearest_position_state(1, data_set[0], data_set, 300, len(data_set))
    first_particle_state = data_set[first_index]
    second_index = nearest_position_state(2, data_set[0], data_set, 300, len(data_set))
    second_particle_state = data_set[second_index]
    third_index = nearest_position_state(3, data_set[0], data_set, 300, len(data_set))
    third_particle_state = data_set[third_index]
    loss = nearest_position(1, data_set[0], first_particle_state) + nearest_position(2, data_set[0],
                                                                                     second_particle_state) + nearest_position(
        3, data_set[0], third_particle_state)

    print(" ")
    loss_values.append(loss.item())

    print(loss)
    input_vec.retain_grad()
    #loss.retain_grad()
    loss.backward()
    print(input_vec.grad)
    # Updates input vector
    with torch.no_grad():
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

    particle1.set_data(data_set[:, 0], data_set[:, 1])
    particle2.set_data(data_set[:, 3], data_set[:, 4])
    particle3.set_data(data_set[:, 6], data_set[:, 7])

    first_text.set_text(f"First Particle Index: {first_index}")
    second_text.set_text(f"Second Particle Index: {second_index}")
    third_text.set_text(f"Third Particle Index: {third_index}")

    bottom.plot([x for x in range(len(loss_values))], loss_values, color="red")

    print(f"Epoch:{i}")
    print(" ")
    return particle1, particle2, particle3, first_text, second_text, third_text


a = 8
b = .0001


writer = animation.FFMpegWriter(fps=int(a/5))
ani = animation.FuncAnimation(figure, update, frames=a, fargs=(b, vec))
ani.save(r"D:\Aki\Pycharm\PycharmProjects\PeriodicThreeBodies\Videos\OtherInitialConds\a2.mp4", writer=writer)



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


# [ 1.5204e+00,  7.6287e-01, -7.2413e-02, -1.7299e-01,  4.2930e-01,
#         -9.4932e-02, -6.4748e-01, -3.9216e-01, -3.2655e-02,  7.0786e-01,
#          5.1624e-01, -1.6740e-04,  2.0788e-01,  3.1259e-01, -2.9388e-02,
#         -8.1860e-01, -7.8648e-01,  2.8100e-02,  3.5703e+01,  3.5734e+01,
#          3.5692e+01]

# [ 1.5305e+00,  7.2942e-01, -7.9352e-02, -1.7411e-01,  4.6347e-01,
#         -7.3587e-02, -6.5644e-01, -3.9288e-01, -4.7061e-02,  6.4761e-01,
#          5.2369e-01, -6.4135e-03,  1.9012e-01,  2.8642e-01, -7.6145e-03,
#         -7.9527e-01, -7.8276e-01,  1.3103e-02,  3.5702e+01,  3.5733e+01,
#          3.5693e+01]