
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy, itertools
from collections import namedtuple
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# いろいろの初期化とか
vector = namedtuple("vector", ["x", "y", "z"])
planet = namedtuple("planet", ["x", "y", "z", "m", "v", "color"])  # (m), (m), (m), vector(x, y, z), color
planets = []
x = []
y = []
z = []
m = []
v = []
colors = []

# シミュレーション設定
delta_t = 5.0 * 10**6
interval_update = 1
size_marker = 5
draw_line = True
range_xyz = 1.0 * 10**2  # 表示範囲(±これになる)

for _ in range(10):
    planets.append(planet(
        np.random.randint(-range_xyz, range_xyz),
        np.random.randint(-range_xyz, range_xyz),
        np.random.randint(-range_xyz, range_xyz),
        np.random.randint(1, 10) * 10**(np.random.randint(2, 4)),
        vector(0, 0, 0),
        "red"
    ))

# 入力データの展開
num = -1
delta_t /= 10**3
n_planets = len(planets)
for p in planets:
    x.append(p.x)
    y.append(p.y)
    z.append(p.z)
    m.append(p.m)
    v.append(p.v)
    x[-1], y[-1], z[-1] = x[-1] - v[-1].x * delta_t, y[-1] - v[-1].y * delta_t, z[-1] - v[-1].z * delta_t

def update(_):
    global num
    num += 1
    for n in range(n_planets):
        x[n], y[n], z[n] = x[n] + v[n].x * delta_t, y[n] + v[n].y * delta_t, z[n] + v[n].z * delta_t
    for i, j in itertools.combinations(range(n_planets), 2):
        x2 = (x[i] - x[j])**2
        y2 = (y[i] - y[j])**2
        z2 = (z[i] - z[j])**2
        if (x2py2pz2 := x2 + y2 + z2) == 0:
            continue
        fmm = scipy.constants.G * delta_t / x2py2pz2
        hogex = 1 if x[i] < x[j] else -1
        hogey = 1 if y[i] < y[j] else -1
        hogez = 1 if z[i] < z[j] else -1
        v[i] = vector(
            v[i].x + fmm * m[j] * x2 / x2py2pz2 * hogex,
            v[i].y + fmm * m[j] * y2 / x2py2pz2 * hogey,
            v[i].z + fmm * m[j] * z2 / x2py2pz2 * hogez
        )
        v[j] = vector(
            v[j].x - fmm * m[i] * x2 / x2py2pz2 * hogex,
            v[j].y - fmm * m[i] * y2 / x2py2pz2 * hogey,
            v[j].z - fmm * m[i] * z2 / x2py2pz2 * hogez
        )
    if not draw_line:
        ax.cla()
    ax.set_aspect("equal")
    ax.scatter(x, y, z, c=colors, s=size_marker)
    plt.title(f"{n_planets}体問題のシミュ (t={delta_t*num}s, dt={delta_t}s)", fontname="MS Gothic")
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True)
    ax.set_xlim(-range_xyz, range_xyz)
    ax.set_ylim(-range_xyz, range_xyz)
    ax.set_zlim(-range_xyz, range_xyz)

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(projection='3d')
ani = animation.FuncAnimation(fig, update, interval=interval_update, frames=1000)
plt.show()
