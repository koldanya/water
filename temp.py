from math import exp, pow, sin, pi
from IPython.display import HTML
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy import sinc
from tqdm import tqdm_notebook as tqdm

def distribution1(M, x, y, x_avr, y_avr, x_scale, y_scale):
    return exp(-pow(((x-x_avr)/x_scale), M))*exp(-pow(((y-y_avr)/y_scale), M))


def distribution2(M, x, x_avr, x_scale):
    return exp(-pow(((x-x_avr)/x_scale), M))


def distribution3(x, y, x_avr, y_avr, x_scale, y_scale):
    return sinc((x-x_avr)/x_scale)*sinc((y-y_avr)/y_scale)
# global_parameters

total_time = 1
dt = 0.01
time = 0
X1 = 0
X2 = 10
dx = 0.05
Y1 = 0
Y2 = 2
dy = 0.05

Nt = int(total_time/dt)+1
print("Шагов по времени = ", Nt)

DX = X2-X1
Nx = int(DX/dx)+1
print("Рзмер по X = ", DX)
print("Максимально узлов по X = ", Nx)

DY = Y2-Y1
Ny = int(DY/dy)+1
print("Рзмер по Y = ", DY)
print("Максимально узлов по Y = ", Ny)
# nodes initialization

x = np.linspace(X1, X2, Nx)
y = np.linspace(Y1, Y2, Ny)
xx, yy = np.meshgrid(x, y)

b = np.zeros((3, Ny, Nx))

h = np.ones((7, Ny, Nx))
u = np.zeros((7, Ny, Nx))
v = np.zeros((7, Ny, Nx))

R = np.zeros((7, Ny, Nx))
Q = np.zeros((7, Ny, Nx))
V = np.zeros((7, Ny, Nx))

lR = np.zeros((Ny, Nx))
lQ = np.zeros((Ny, Nx))
lV = np.zeros((Ny, Nx))


# initial condition

for j in range(Ny):
    for i in range(Nx):
        h[0, j, i] /= 2
        h[0, j, i] += distribution1(2, x[i], y[j], DX*4/5, DY*1/2, DY*1/6, DY*1/6)
        h[0, j, i] += distribution1(2, x[i], y[j], DX*1/5, DY*1/2, DY*1/6, DY*1/6)
        h[0, j, i] += distribution2(50, x[i], DX*1/2, DX*1/12)
#         h[0, j, i] += distribution3(x[i], y[j], DX*4/5, DY*1/2, DY*2/18, DY*2/18)
# plots initialization

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(18, 12))
frames = []
ax0.set_xlim(X1, X2)
ax0.set_ylim(Y1, Y2)
ax1.set_xlim(X1, X2)
ax1.set_ylim(Y1, Y2)
ax2.set_xlim(X1, X2)
ax2.set_ylim(0, 2)
ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
ax0.set_title('Высота поверхности')
ax1.set_title('Поле скоростей')
ax2.set_title('Распределение высот на слое y=1')
plt.close()

# t=0
im0 = ax0.pcolormesh(x, y, h[0, :, :], cmap='coolwarm', vmin=0, vmax=2)
fig.colorbar(im0, ax=ax0)
im1 = ax1.quiver(x, y, u[0, :, :], -v[0, :, :],
                 units='xy', angles='xy', width=0.01)
ax1.quiverkey(im1, 0.85, 0.65, 1, r'$1 \frac{m}{s}$',  fontproperties={
              'size': 24}, labelpos='W', coordinates='figure')
im2, = ax2.plot(x, h[0, int(Ny/2), :], 'r')
im3, = ax2.plot(x, b[0, int(Ny/2), :], 'b')
frames.append([im0, im1, im2, im3])
# Computation 

for i in tqdm(range(2, 11, 4)):
    print(i)
    u[0, :, :] = np.sin((xx**2/(11-i))*np.pi)
    v[0, :, :] = np.cos(yy*np.pi)
    h[0, :, :] = 1 + u[0, :, ::-1] * v[0, :, :]

    im0 = ax0.pcolormesh(x, y, h[0, :, :], cmap='twilight', vmin=0, vmax=2)
    im1 = ax1.quiver(x, y, u[0, :, :], v[0, :, :], units='xy', angles='xy', width=0.01)
    im2, = ax2.plot(x, h[0, 1, :], 'r')
    im3, = ax2.plot(x, b[0, 1, :], 'b')
    frames.append([im0, im1, im2, im3])
plt.show()
# showing and saving animation
ani = animation.ArtistAnimation(fig,frames, interval=33.33, blit=True)
ani.save('animation2.gif', writer='pillow', fps=30)
#ani.save('animation.mp4', writer='ffmpeg', fps=30)
#HTML(ani.to_jshtml(fps=30))