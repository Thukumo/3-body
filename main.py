
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy, itertools
from collections import namedtuple
import numpy as np

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
#delta_t = 1.0*10**7
delta_t = 5.0*10**(6)
interval_update = 1
size_marker = 5
draw_line = True
range_xy = 1.0 * 10**2 #表示範囲(±これになる)
"""
planets.append(planet(0, 0, 5.972*10**3, vector(0, 0), "red"))
planets.append(planet(0, 4.0*10**1, 7.0*10**2, vector(0.05, 0), "blue"))
"""
#"""
planets.append(planet(-6.0*10**1, -4.0*10**1, 5.972*10**3, vector(0, 0), "red"))
planets.append(planet(6.0*10**1, 4.0*10**1, 7.0*10**2, vector(0, 0), "blue"))
planets.append(planet(6.0*10**1, -4.0*10**1, 7.0*10**2, vector(0, 0), "green"))
"""
for _ in range(10):
    #planets.append(planet(np.random.randint(-range_xy, range_xy), np.random.randint(-range_xy, range_xy), np.random.randint(1, 10)*10**(-1), vector(np.random.choice([-1, 1])*np.random.rand(), np.random.choice([-1, 1])*np.random.rand()), "red"))
    planets.append(planet(np.random.randint(-range_xy, range_xy), np.random.randint(-range_xy, range_xy), np.random.randint(1, 10)*10**(np.random.randint(2, 4)), vector(0, 0), "red"))
"""
#入力データの展開
num = -1
delta_t /= 10**3
n_planets = len(planets)
for n in range(n_planets):
    x.append(planets[n].x)
    y.append(planets[n].y)
    m.append(planets[n].m)
    v.append(planets[n].v)
    colors.append(planets[n].color)
    x[n], y[n] = x[n] - v[n].x*delta_t, y[n] - v[n].y*delta_t

def update(_):
    global num
    #print(v)
    num+=1
    for n in range(n_planets): x[n], y[n] = x[n] + v[n].x*delta_t, y[n] + v[n].y*delta_t
    for i, j in itertools.combinations(range(n_planets), 2):
        fmm = scipy.constants.G*delta_t/((x[i]-x[j])**2 + (y[i]-y[j])**2)
        x2 = (x[i] - x[j])**2
        y2 = (y[i] - y[j])**2
        if x2+y2 == 0:
            continue
        hogex = 1 if x[i] < x[j] else -1 # == 0のときはx2が0となるので考えなくてよい
        hogey = 1 if y[i] < y[j] else -1
        v[i] = vector(v[i].x + fmm*m[j]*x2/(x2+y2)*hogex, v[i].y + fmm*m[j]*y2/(x2+y2)*hogey)
        v[j] = vector(v[j].x - fmm*m[i]*x2/(x2+y2)*hogex, v[j].y - fmm*m[i]*y2/(x2+y2)*hogey)
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
