import cmath
from matplotlib import pyplot as plt
from imageio import imread
import numpy as np
from PIL import Image, ImageDraw

i = cmath.sqrt(-1)

def f(c, z):
    return z**2 + c

def iterate_f(c, x, n):
    xNew = x
    for iterate in range(n):
        xNew = f(c, xNew)
    return xNew

def plot_orbit_f(c, z, NumberOfIterates, scale, DrawCircle):

    # Plots the orbit of f_c on the dynamical plane
    #
    # Inputs:
    # c = The parameter, so that we are iterating z -> z^2 + c. Must be inputted as a + b*i, for real a,b
    # z = Initial point, must be inputted as above
    # NumberOfIterates = Integer number of iterates to plot
    # scale = 4-tuple, say (a, b, c, d), so that the plot shows the plane for a < Re z < b, c < Im z < d
    #
    # Outputs:
    # Pyplot should splash an image on to your screen, showing the plot
    
    RealPart = []
    ImagPart = []
    point = z
    for j in range(NumberOfIterates):
        point = f(c, point)
        RealPart.append(point.real)
        ImagPart.append(point.imag)
    plt.axhline(0)
    plt.axvline(0)
    plt.plot(RealPart, ImagPart, 'C3', lw=3)
    if DrawCircle:
        theta = np.linspace(-np.pi, np.pi, 200)
        plt.plot(np.sin(theta)*2, np.cos(theta)*2)
    plt.scatter(RealPart, ImagPart, s=120)
    plt.xlim(scale[0], scale[1])
    plt.ylim(scale[2], scale[3])
    plt.xlabel('Re z')
    plt.ylabel('Im z')
    plt.axes().set_aspect(1)
    plt.show()

def mandelbrot(c, NumberOfIterates):
    n = 0
    z = 0
    while abs(z) <= 2 and n < NumberOfIterates:
        n = n + 1
        z = f(c, z)
    if n == NumberOfIterates:
        IsInMandelbrot = True
    else:
        IsInMandelbrot = 'Unknown'
    return (IsInMandelbrot, n)

def generate_mandelbrot_png(NumberOfIterates, scale, ImageWidth, filename):

    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x in range(0, ImageWidth):
        for y in range(0, ImageHeight):
            c = scale[0] + (x / ImageWidth) * (scale[1] - scale[0]) + (scale[2] + (y / ImageHeight) * (scale[3] - scale[2])) * i
            m = mandelbrot(c, NumberOfIterates)[1]
            ColourNum = 255 - int(m * 255 / NumberOfIterates)
            draw.point([x, y], (ColourNum, ColourNum, ColourNum))
    im.save(filename + '.png', 'PNG')

def plot_image_on_plane(PlaneScale, ImageScale, filename):

    # Plots an image on the complex plane, with added axes on border
    #
    # Inputs:
    # PlaneScale = 4-tuple specifying the part of the complex plane - order goes [top, bottom, left, right]
    # ImageScale = 4-tuple specifying where image goes on plane - order goes [top, bottom, left, right]
    
    img = imread(filename)
    plt.xlim(PlaneScale[0], PlaneScale[1])
    plt.ylim(PlaneScale[2], PlaneScale[3])
    plt.xticks(np.arange(-0.18, -0.13, 0.01))
    plt.yticks(np.arange(1.02, 1.05, 0.01))
    plt.xlabel('Re z')
    plt.ylabel('Im z')
    plt.axes().set_aspect(1)
    plt.imshow(img, zorder=0, extent=[ImageScale[0], ImageScale[1], ImageScale[2], ImageScale[3]])
    plt.show()
