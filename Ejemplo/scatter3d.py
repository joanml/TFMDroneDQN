# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import math

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
'''for m, zlow, zhigh in [('o', 10, -25), ('^', -30, -5)]:
    print( m, zlow, zhigh)
    xs = randrange(n, 50, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, 0, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')'''
x= list()
y= list()
for m in range(1,100):
    # print(m[1],m[0])
    x.append(m*2)
    y.append(m%2)

x = np.asarray(x)
y = np.asarray(y)
fig = Figure()
canvas = FigureCanvas(fig)
ax = fig.gca()
ax.text(0.0,0.0,"Test", fontsize=45)
ax.axis('off')

canvas.draw()
plt.show()
#image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')


'''
plt.scatter(x, y)
plt.plot(x, y, '-o')
plt.show()'''
