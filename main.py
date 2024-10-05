
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy, itertools
from collections import namedtuple
import numpy as np
import scipy.constants
from math import sqrt
from tqdm import tqdm

#いろいろの初期化とか
vector = namedtuple("vector", ["x", "y"])
planet = namedtuple("planet", ["x", "y", "m", "v", "color"]) #(m), (m), vector(x, y), color
planets, x, y, m, v, colors = [], [], [], [], [], []

# シミュレーション設定
#delta_t = 1.0e7
interval_update = 1
"""
range_xy = 1.0e2
planets.append(planet(0, 0, 5.972e3, vector(0, 0), "red"))
planets.append(planet(0, 4.0e1, 7.0e2, vector(0.05, 0), "blue"))
"""
delta_t = 5.0e3
range_xy = 1.0e2
size_marker = 25
draw_line = 0
plot = 0
frame = 1200
fps = 40
planets.append(planet(-6.0e1, -4.0e1, 5.972e3, vector(0, 0), "red"))
planets.append(planet(6.0e1, 4.0e1, 7.0e2, vector(0, 0), "blue"))
planets.append(planet(6.0e1, -4.0e1, 7.0e2, vector(0, 0), "green"))

"""
delta_t = 5.0e2
range_xy = 1.5
size_marker = 15
draw_line = False
planets.append(planet(0, 0, 100, vector(0, 0), "red"))
planets.append(planet(0, 1, 1.0e-2, vector(-sqrt(scipy.constants.G*100), 0), "blue"))
#planets.append(planet(1, 0, 1.0e-2, vector(0, -sqrt(scipy.constants.G*100)), "green"))
"""
"""
range_xy = 2e10
delta_t = 8.64e4*4
draw_line = False
situryo = 5.9724e25
planets.append(planet(-0.1e11, -0.01e11, situryo, vector(0, 1.0e2), "red"))
planets.append(planet(0.1e11, 0.01e11, situryo, vector(0, -1.0e2), "blue"))
"""
"""
delta_t = 2.0e3
range_xy = 1.0e2
size_marker = 5
draw_line = 1
for _ in range(3): planets.append(planet(np.random.randint(-range_xy, range_xy), np.random.randint(-range_xy, range_xy), np.random.randint(1, 10) * 10 ** np.random.randint(2, 4), vector(0, 0), "red"))
"""






#入力データの展開
num = -1
n_planets = len(planets)
for p in planets:
    x.append(p.x)
    y.append(p.y)
    m.append(p.m)
    v.append(p.v)
    colors.append(p.color)
    x[-1], y[-1] = x[-1] - v[-1].x*delta_t, y[-1] - v[-1].y*delta_t
del planets
oldpoint = None

def update(_):
    global num, oldpoint
    num+=1
    for n in range(n_planets): x[n], y[n] = x[n] + v[n].x*delta_t, y[n] + v[n].y*delta_t
    for i, j in itertools.combinations(range(n_planets), 2):
        xdiff = x[i] - x[j]
        ydiff = y[i] - y[j]
        if (x2py2 := xdiff**2 + ydiff**2) == 0: continue
        x2py2sqrt = sqrt(x2py2)
        cos, sin = xdiff/x2py2sqrt, ydiff/x2py2sqrt
        fmm = scipy.constants.G*delta_t/x2py2
        v[i], v[j] = vector(v[i].x - fmm*m[j]*cos, v[i].y - fmm*m[j]*sin), vector(v[j].x + fmm*m[i]*cos, v[j].y + fmm*m[i]*sin)
    if (not draw_line) and oldpoint != None : oldpoint.remove()
    ax.set_aspect("equal")
    oldpoint = plt.scatter(x, y, c=colors, s=size_marker)
    plt.title(f"{n_planets}体問題のシミュ (t={delta_t*num}s, dt={delta_t}s)", fontname="MS Gothic")

fig = plt.figure()
ax = plt.axes()
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.grid(True)
plt.xlim(-range_xy, range_xy)
plt.ylim(-range_xy, range_xy)
if "frame" in globals(): ani = animation.FuncAnimation(fig, update, interval=interval_update, frames=tqdm(range(frame)))
else: ani = animation.FuncAnimation(fig, update, interval=interval_update)
if plot:
    plt.show()
else:
    ani.save("test.gif", writer = 'imagemagick', fps=fps)
