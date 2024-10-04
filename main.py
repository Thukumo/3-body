
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy, itertools
from collections import namedtuple
import numpy as np
import scipy.constants
from math import sqrt

#いろいろの初期化とか
vector = namedtuple("vector", ["x", "y"])
planet = namedtuple("planet", ["x", "y", "m", "v", "color"]) #(m), (m), vector(x, y), color
planets = []
x = []
y = []
m = []
v = []
colors = []

# シミュレーション設定
#delta_t = 1.0e7
interval_update = 1
"""
range_xy = 1.0e2
planets.append(planet(0, 0, 5.972e3, vector(0, 0), "red"))
planets.append(planet(0, 4.0e1, 7.0e2, vector(0.05, 0), "blue"))
"""
"""
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
draw_line = False
for _ in range(100): planets.append(planet(np.random.randint(-range_xy, range_xy), np.random.randint(-range_xy, range_xy), np.random.randint(1, 10) * 10 ** np.random.randint(2, 4), vector(0, 0), "red"))
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

def update(_):
    global num
    #print(v)
    num+=1
    for n in range(n_planets): x[n], y[n] = x[n] + v[n].x*delta_t, y[n] + v[n].y*delta_t
    for i, j in itertools.combinations(range(n_planets), 2):
        x2 = (x[i] - x[j])**2
        y2 = (y[i] - y[j])**2
        if (x2py2 := x2 + y2) == 0: continue
        ucos = sqrt(x2/x2py2)
        usin = sqrt(y2/x2py2)
        fmm = scipy.constants.G*delta_t/x2py2
        hogex = 1 if x[i] < x[j] else -1 # == 0のときはx2が0となるので考えなくてよい
        hogey = 1 if y[i] < y[j] else -1
        v[i] = vector(v[i].x + fmm*m[j]*ucos*hogex, v[i].y + fmm*m[j]*usin*hogey)
        v[j] = vector(v[j].x - fmm*m[i]*ucos*hogex, v[j].y - fmm*m[i]*usin*hogey)
    if not draw_line: plt.cla()
    ax.set_aspect("equal")
    plt.scatter(x, y, c=colors, s=size_marker)
    plt.title(f"{n_planets}体問題のシミュ (t={delta_t*num}s, dt={delta_t}s)", fontname="MS Gothic")
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.xlim(-range_xy, range_xy)
    plt.ylim(-range_xy, range_xy)

fig = plt.figure()
ax = plt.axes()
ani = animation.FuncAnimation(fig, update, interval=interval_update, frames=1000)
#ani.save("test.gif", writer = 'imagemagick')
# グラフを表示
plt.show()
