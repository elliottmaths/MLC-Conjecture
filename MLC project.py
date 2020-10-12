import cmath
from matplotlib import pyplot as plt
from imageio import imread
import matplotlib.cbook as cbook
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import random
from math import floor

# In this module, i is reserved for the square root of -1
# Complex numbers are thus written as a + b*i, for some numbers a, b
i = cmath.sqrt(-1)

def f(c, z):

    # Calculates the value f_c(z)
    #
    # Inputs:
    # c = complex number that is a parameter that defines the quadratic f_c
    # z = complex number that is the input to the function
    #
    # Outputs:
    # returns the value f_c(z)
    
    return z**2 + c

def iterate_f(c, z, n):

    # Calculates the nth iterate of f_c
    #
    # Inputs:
    # c = complex number that is a parameter that defines the quadratic f_c
    # z = complex number that is the initial input to the function
    # n = integer specifying how many times to iterate
    #
    # Outputs:
    # zNew = the value f_c^{\circ n} (z)
    
    zNew = z
    for iterate in range(n):
        zNew = f(c, zNew)
    return zNew

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
    # pyplot will splash an image on-screen
    
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

#plot_orbit_f(-0.7-0.3*i, 0, 30, (-8, 2.5, -3.5, 6.5), True)

def generate_mandelbrot_png(NumberOfIterates, scale, ImageWidth, filename):

    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x in range(0, ImageWidth):
        for y in range(0, ImageHeight):
            c = scale[0] + (x / ImageWidth) * (scale[1] - scale[0]) + (scale[2] + (y / ImageHeight) * (scale[3] - scale[2])) * i
            n = 0
            z = 0
            while abs(z) <= 2 and n < NumberOfIterates:
                n = n + 1
                z = f(c, z)
            ColourNum = 255 - int(n * 255 / NumberOfIterates)
            draw.point([x, y], (ColourNum, ColourNum, ColourNum))
    im.save(filename + '.png', 'PNG')

#generate_mandelbrot_png(150, [-2, 1, -1.5, 1.5], 500, "zoot")

def plot_image_on_plane(PlaneScale, ImageScale, filename, *args):

    # Plots an image on the complex plane, with added axes on border
    #
    # Inputs:
    # PlaneScale = 4-tuple specifying the part of the complex plane - order goes [left, right, top, bottom]
    # ImageScale = 4-tuple specifying where image goes on plane - order goes [left, right, top, bottom]
    # filename = string that is the filename (including extension) of the image, which must be in the current working directory
    # *args = optional input, if you want to specify the numbers on the axes, you can write 6 additional inputs, and we will have x-axis numbers from args[0] to args[1] in steps of length args[2], and y-axis numbers from args[3] to args[4] in steps of length args[5]
    #
    # Outputs:
    # pyplot will splash an image on-screen
    
    img = imread(filename)
    plt.xlim(PlaneScale[0], PlaneScale[1])
    plt.ylim(PlaneScale[2], PlaneScale[3])
    if len(args) != 0:
        plt.xticks(np.arange(args[0], args[1], args[2]))
        plt.yticks(np.arange(args[3], args[4], args[5]))
    plt.xlabel('Re z')
    plt.ylabel('Im z')
    plt.axes().set_aspect(1)
    plt.imshow(img, zorder=0, extent=[ImageScale[0], ImageScale[1], ImageScale[2], ImageScale[3]])
    plt.show()

#plot_image_on_plane([-0.18, -0.14, 1.013, 1.053], [-0.18, -0.14, 1.013, 1.053], 'mandelbrot zoom.png')

def convert_complex_to_img_coord(z, scale, ImageWidth, ImageHeight):

    # Takes a complex number and calculates its coordinate as a PIL image, according to the part of the complex plane pictured (scale), and the number of pixels in the image (ImageWidth, ImageHeight)
    #
    # Inputs:
    # z = complex number whose image coordinates we will calculate
    # scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, top, bottom]
    # ImageWidth = number of pixels that the image has for its width
    # ImageHeight = number of pixels that the image has for its height
    #
    # Outputs:
    # returns a 2-tuple that gives us image coordinates. According to PIL's logic, the 1st entry is how far along we are from the left, and 2nd is how far down from the top
    
    HowFarAlong = (z.real - scale[2]) / abs(scale[3] - scale[2])
    HowFarDown = abs((z.imag - scale[1])) / abs(scale[1] - scale[0])
    return (floor(HowFarAlong * ImageWidth), floor(HowFarDown * ImageHeight))

def convert_img_coord_to_complex(x, y, scale, ImageWidth, ImageHeight):

    # Converts PIL image coordinates to a complex number, according to which part of the complex plane we are on and the size of the image
    #
    # Inputs:
    # x = integer that says how many pixels we are from the left of the image
    # y = integer that says how many pixels we are from the top of the image
    # scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, top, bottom]
    # ImageWidth = number of pixels that the image has for its width
    # ImageHeight = number of pixels that the image has for its height
    #
    # Outputs:
    # returns a complex number, corresponding to the image coordinates in the plane
    
    HowFarAlong = x / ImageWidth
    RealPart = scale[0] + HowFarAlong * abs(scale[1] - scale[0])
    HowFarDown = y / ImageHeight
    ImagPart = scale[3] - HowFarDown * abs(scale[3] - scale[2])
    return RealPart + ImagPart*i

def generate_julia_png(c, NumberOfIterates, scale, ImageWidth, filename):

    # Generates an image of the Julia set of a point in the parameter plane
    #
    # Inputs:
    # c = complex number that is a point in the parameter plane, so that we are considering the Julia set of f_c
    # NumberOfIterates = integer that specifies how many iterates we calculate - a higher number takes more time but gives a more accurate depiction
    # scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, top, bottom]
    # ImageWidth = integer that gives the desired number of pixels that the image has across - height will be calculated accordingly, based on scale
    # filename = string that the image of the Julia set will be saved as - note, this should not include the file extension, e.g. .png
    #
    # Outputs:
    # saves a .png image in the current working directory, depicting the Julia set of f_c
    #
    # NOTE: In contrast to alt_generate_julia_png, this function uses the inverse iteration algorithm
    # Accordingly, it is more efficient than the other version, but gives less detail for certain Julia sets
    
    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    LastPoint = 1
    for iterate in range(NumberOfIterates):
        which = random.choice([True,False])
        if which:
            NextPoint = cmath.sqrt(LastPoint - c)
        else:
            NextPoint = -1 * cmath.sqrt(LastPoint - c)
        coords = convert_complex_to_img_coord(NextPoint, scale, ImageWidth, ImageHeight)
        draw.point([coords[0], coords[1]], (0, 0, 0))
        LastPoint = NextPoint
    im.save(filename + '.png', 'PNG')

def distance(a,b):

    # Calculates the Euclidean distance between two points in the complex plane
    #
    # Inputs:
    # a = complex number that is one of the points under consideration
    # b = complex number that is the other point under consideration
    #
    # Outputs:
    # returns a float that gives the distance between a and b
    
    return cmath.sqrt((a.real - b.real)**2 + (a.imag - b.imag)**2)

def alt_generate_julia_png(c, NumberOfIterates, scale, ImageWidth, filename):

    # Generates an image of the Julia set of a point in the parameter plane
    #
    # Inputs:
    # c = complex number that is a point in the parameter plane, so that we are considering the Julia set of f_c
    # NumberOfIterates = integer that specifies how many iterates we calculate - a higher number takes more time but gives a more accurate depiction
    # scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, top, bottom]
    # ImageWidth = integer that gives the desired number of pixels that the image has across - height will be calculated accordingly, based on scale
    # filename = string that the image of the Julia set will be saved as - note, this should not include the file extension, e.g. .png
    #
    # Outputs:
    # saves a .png image in the current working directory, depicting the Julia set of f_c
    #
    # NOTE: In contrast to generate_julia_png, this function uses the so-called boundary scanning method
    # Accordingly, it is less efficient than the other version, but gives more detail for certain Julia sets
    
    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for height in range(ImageHeight):
        for width in range(ImageWidth):
            BoolList = []
            for corner1 in [0,1]:
                for corner2 in [0,1]:
                    CornerHeight = 2*height + corner1
                    CornerWidth = 2*width + corner2
                    point = convert_img_coord_to_complex(CornerWidth, CornerHeight, scale, 2*ImageWidth, 2*ImageHeight)
                    x = point
                    count = 0
                    while count < NumberOfIterates and abs(x) < 3:
                        x = f(c, x)
                        count = count + 1
                    if abs(x) >= 3:
                        BoolList.append(True)
                    else:
                        BoolList.append(False)
            if BoolList != [True, True, True, True] and BoolList != [False, False, False, False]:
                draw.point([height, width], (0, 0, 0))
    im = im.rotate(90)
    im = ImageOps.flip(im)
    im.save(filename + '.png', 'PNG')

#generate_julia_png(0, 10000, [-1.5, 1.5, -1.5, 1.5], 1200, "circle")
#alt_generate_julia_png(-0.12+0.75*i, 200, [-1.5, 1.5, -1.5, 1.5], 1200, "DouadyRabbit")
#alt_generate_julia_png(-0.7-0.3*i, 200, [-1.5, 1.5, -1.5, 1.5], 400, "TotallyDis")
