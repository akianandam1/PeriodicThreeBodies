from TorchNumericalSolver import torchIntegrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import torch
import time

mpl.rcParams['animation.ffmpeg_path'] = r'D:\Aki\Python37\Lib\site-packages\ffmpeg-5.0-essentials_build\bin\ffmpeg.exe'


# input_vec = torch.tensor([2.3050e-01, -4.0595e-02, 8.5479e-44])
# full_vec = torch.cat(
#     [torch.tensor([1, 0, 0, 0, 0, 0, 0, 1, 0]), input_vec, torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])])
full_vec = torch.tensor([-7.7959e-01,  2.1395e-01,  0.0000e+00,  9.8176e-02, -4.1617e-02,
         0.0000e+00,  6.8141e-01, -1.7234e-01,  0.0000e+00, -2.0357e-02,
         9.7143e-02,  0.0000e+00,  2.7091e-01,  5.8770e-01,  0.0000e+00,
        -2.5055e-01, -6.8485e-01,  0.0000e+00,  3.5007e+01,  3.5007e+01,
         3.5007e+01])
start = time.time()

sol = torchIntegrate(full_vec, .001, 10).numpy()
end = time.time()

calc  = end-start

first = time.time()
r1_sol = sol[:, 0:3]
r2_sol = sol[:, 3:6]
r3_sol = sol[:, 6:9]


# INITIAL POSITIONS: (-1,0), (1,0), (0,0)
#
# INITIAL VELOCITIES: (p1,p2), (p1,p2), (-2p1,-2p2)
#
# p1: 0.347111
#
# p2: 0.532728



# Create figure
fig = plt.figure(figsize=(20, 20))  # Create 3D axes
ax = fig.add_subplot(111, projection="3d")  # Plot the orbits
ax.set_zlim(-3, 3)
ax.set_title("Visualization of orbits of stars in a three body system\n", fontsize=28)

particle1, = plt.plot([], [], [], color='r')
particle2, = plt.plot([], [], [], color='g')
particle3, = plt.plot([], [], [], color='b')
plt.xlim(-3, 3)
plt.ylim(-3, 3)

p1, = plt.plot([], [], marker='o', color='r', label="Real first star")
p2, = plt.plot([], [], marker='o', color='g', label="Real second star")
p3, = plt.plot([], [], marker='o', color='b', label="Real third star")


# model_particle1, = plt.plot([], [], [], color='r')
# model_particle2, = plt.plot([], [], [], color='g')
# model_particle3, = plt.plot([], [], [], color='b')
#
# model_p1, = plt.plot([], [], marker='s', color='r', label="Model's first star")
# model_p2, = plt.plot([], [], marker='s', color='g', label="Model's second star")
# model_p3, = plt.plot([], [], marker='s', color='b', label="Model's third star")

ax.legend(loc="upper left", fontsize=28)
def update(i):
    i*=10
    particle1.set_data(r1_sol[:i, 0], r1_sol[:i, 1])
    particle1.set_3d_properties(r1_sol[:i, 2])
    particle2.set_data(r2_sol[:i, 0], r2_sol[:i, 1])
    particle2.set_3d_properties(r2_sol[:i, 2])
    particle3.set_data(r3_sol[:i, 0], r3_sol[:i, 1])
    particle3.set_3d_properties(r3_sol[:i, 2])

    # p1.set_data(r1_sol[i:i + 1, 0], r1_sol[i:i + 1, 1])
    # p1.set_3d_properties(r1_sol[i:i + 1, 2])
    # p2.set_data(r2_sol[i:i + 1, 0], r2_sol[i:i + 1, 1])
    # p2.set_3d_properties(r2_sol[i:i + 1, 2])
    # p3.set_data(r3_sol[i:i + 1, 0], r3_sol[i:i + 1, 1])
    # p3.set_3d_properties(r3_sol[i:i + 1, 2])
    print(i)

    # model_particle1.set_data(model_r1_sol[:i, 0], model_r1_sol[:i, 1])
    # model_particle1.set_3d_properties(model_r1_sol[:i, 2])
    # model_particle2.set_data(model_r2_sol[:i, 0], model_r2_sol[:i, 1])
    # model_particle2.set_3d_properties(model_r2_sol[:i, 2])
    # model_particle3.set_data(model_r3_sol[:i, 0], model_r3_sol[:i, 1])
    # model_particle3.set_3d_properties(model_r3_sol[:i, 2])
    #
    # model_p1.set_data(model_r1_sol[i:i + 1, 0], model_r1_sol[i:i + 1, 1])
    # model_p1.set_3d_properties(model_r1_sol[i:i + 1, 2])
    # model_p2.set_data(model_r2_sol[i:i + 1, 0], model_r2_sol[i:i + 1, 1])
    # model_p2.set_3d_properties(model_r2_sol[i:i + 1, 2])
    # model_p3.set_data(model_r3_sol[i:i + 1, 0], model_r3_sol[i:i + 1, 1])
    # model_p3.set_3d_properties(model_r3_sol[i:i + 1, 2])

    return particle1, particle2, particle3, p1, p2, p3, #model_particle1, model_particle2, model_particle3, model_p1, model_p2, model_p3


second = time.time()
plot_time = second-first

writer = animation.FFMpegWriter(fps=100)
ani = animation.FuncAnimation(fig, update, frames=1_000, interval=25, blit=True)
ani.save(r"D:\Aki\Pycharm\PycharmProjects\PeriodicThreeBodies\Videos\April11\a1.mp4", writer=writer)
second = time.time()
print(f"Calculation: {calc}")
print(f"Plotting: {plot_time}")