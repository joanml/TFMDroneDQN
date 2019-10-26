from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

fig = Figure(figsize=(5, 5), dpi=100)
# A canvas must be manually attached to the figure (pyplot would automatically
# do it).  This is done by instantiating the canvas with the figure as
# argument.
canvas = FigureCanvasAgg(fig)

# Do some plotting.
ax = fig.add_subplot(111)
x= list()
y= list()
for m in range(1,100):
    # print(m[1],m[0])
    x.append(m*2)
    y.append(m%2)

x = np.asarray(x)
y = np.asarray(y)
ax.scatter(x,y)
ax.axis('off')

# Option 1: Save the figure to a file; can also be a file-like object (BytesIO,
# etc.).
fig.savefig("test.png")

# Option 2: Save the figure to a string.
canvas.draw()
s, (width, height) = canvas.print_to_buffer()

# Option 2a: Convert to a NumPy array.
X = np.frombuffer(s, np.uint8).reshape((height, width, 4))

# Option 2b: Pass off to PIL.
from PIL import Image
im = Image.frombytes("RGBA", (width, height), s)

# Uncomment this line to display the image using ImageMagick's `display` tool.
im.show()