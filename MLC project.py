import cmath
from matplotlib import pyplot as plt
from imageio import imread
import matplotlib.cbook as cbook
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import random
from math import floor, log

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
    # returns the value f_c(z) = z^2 + c
    
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

#plot_orbit_f(-0.15472 + 1.031046*i, 0, 100, (-2, 2, -2, 2), False)

def generate_mandelbrot_png(NumberOfIterates, scale, ImageWidth, filename):

    # Generates an image of the Mandelbrot set on the parameter plane
    #
    # Inputs:
    # NumberOfIterates = integer that is max number of iterates to which we test whether a point is in the set - a higher value will give a more accurate image, but will take longer
    # scale = 4-tuple specifying which part of the complex plane we want to look at, so that we may e.g. zoom in - order goes [left, right, top, bottom]
    # ImageWidth = integer that is number of pixels that the output image will have as width
    # filename = string that the image will have as its name - file extensions need not be included in this
    #
    # Outputs:
    # will save a .png image in the current working directory, displaying the specified part of the Mandelbrot set

    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x in range(0, ImageWidth):
        for y in range(0, ImageHeight):
            c = scale[0] + (x / ImageWidth) * (scale[1] - scale[0]) + (scale[3] + (y / ImageHeight) * (scale[2] - scale[3])) * i
            n = 0
            z = 0
            while abs(z) <= 2 and n < NumberOfIterates:
                n = n + 1
                z = f(c, z)
            ColourNum = 255 - int(n * 255 / NumberOfIterates)
            draw.point([x, y], (ColourNum, ColourNum, ColourNum))
    im.save(filename + '.png', 'PNG')

#generate_mandelbrot_png(150, [0, 0.5, 0.25, -0.25], 3000, "cusp")

def generate_hyperbolic_components_png(NumberOfComponents, scale, ImageWidth, filename):

    # Generates an image of the main cardioid of the Mandelbrot set on the parameter plane, that is, the parameters that have an attracting fixed point
    #
    # Inputs:
    # NumberOfComponents = integer specifying how many hyperbolic components to generate. Note we are counting e.g. all hyperbolic components with 3-cycles
    #                      as 'one component'. Also, this function is work in progress - can currently only go up to 2 components
    # scale = 4-tuple specifying which part of the complex plane we want to look at, so that we may e.g. zoom in - order goes [left, right, top, bottom]
    # ImageWidth = integer that is number of pixels that the output image will have as width
    # filename = string that the image will have as its name - file extensions need not be included in this
    #
    # Outputs:
    # will save a .png image in the current working directory, displaying the specified part of the main cardioid

    if NumberOfComponents not in [1, 2]:
        raise ValueError("NumberOfComponents must be 1 or 2.")
    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x in range(0, ImageWidth):
        for y in range(0, ImageHeight):
            c = scale[0] + (x / ImageWidth) * (scale[1] - scale[0]) + (scale[3] + (y / ImageHeight) * (scale[2] - scale[3])) * i
            if abs(1-cmath.sqrt(1-4*c)) < 1:
                draw.point([x, y], (0, 0, 0))
            elif abs(4*(c + 1)) < 1:
                if NumberOfComponents > 1:
                    draw.point([x, y], (127, 127, 127))
            else:
                draw.point([x, y], (255, 255, 255))
    im.save(filename + '.png', 'PNG')
    
#generate_hyperbolic_components_png(2, [-2., 1, 1.5, -1.5], 3000, "2components")

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
    if len(args) == 6:
        plt.xticks(np.arange(args[0], args[1], args[2]))
        plt.yticks(np.arange(args[3], args[4], args[5]))
    plt.xlabel('Re z')
    plt.ylabel('Im z')
    plt.axes().set_aspect(1)
    plt.imshow(img, zorder=0, extent=[ImageScale[0], ImageScale[1], ImageScale[2], ImageScale[3]])
    plt.show()

#plot_image_on_plane([-2.2, 1, 1.5, -1.5], [-2, 1, 1.5, -1.5], '2components.png')

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
    
    HowFarAlong = (z.real - scale[0]) / abs(scale[1] - scale[0])
    HowFarDown = (scale[2] - z.imag) / abs(scale[3] - scale[2])
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
    ImagPart = scale[2] - HowFarDown * abs(scale[3] - scale[2])
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
    
    ratio = abs(scale[2] - scale[3]) / abs(scale[1] - scale[0])
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
        if coords[0] > 0 and coords[0] < ImageWidth and coords[1] > 0 and coords[1] < ImageHeight:
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
    
    ratio = abs(scale[2] - scale[3]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for height in range(ImageHeight, 0, -1):
        for width in range(ImageWidth):
            BoolList = []
            for corner1 in [0,1]:
                for corner2 in [0,1]:
                    CornerHeight = 2*height + corner1
                    CornerWidth = 2*width + corner2
                    x = convert_img_coord_to_complex(CornerWidth, CornerHeight, scale, 2*ImageWidth, 2*ImageHeight)
                    count = 0
                    while count < NumberOfIterates and abs(x) < 3:
                        x = f(c, x)
                        count = count + 1
                    if abs(x) >= 3:
                        BoolList.append(True)
                    else:
                        BoolList.append(False)
            if BoolList != [True, True, True, True] and BoolList != [False, False, False, False]:
                draw.point([width, height], (0, 0, 0))
    im.save(filename + '.png', 'PNG')

#generate_julia_png(0, 10000, [-1.5, 1.5, 1.5, -1.5], 10000, "Circle")
#alt_generate_julia_png(-0.12+0.75*i, 200, [-1.5, 1.5, 1.5, -1.5], 1200, "DouadyRabbit")


def generate_cantor_png(NumberOfIterates, scale, ImageWidth, filename):

    # Generates an image of the Cantor middle third set
    #
    # Inputs:
    # NumberOfIterates = integer that specifies how many iterates we calculate - a higher number takes more time but gives a more accurate depiction
    # scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, top, bottom]
    # ImageWidth = integer that gives the desired number of pixels that the image has across - height will be calculated accordingly, based on scale
    # filename = string that the image of the Cantor set will be saved as - note, this should not include the file extension, e.g. .png
    #
    # Outputs:
    # saves a .png image in the current working directory, depicting the Cantor middle third set

    ratio = abs(scale[2] - scale[3]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    Start = convert_complex_to_img_coord(0, scale, ImageWidth, ImageHeight)
    End = convert_complex_to_img_coord(1, scale, ImageWidth, ImageHeight)
    for pixel in range(Start[0], Start[0] + int((End[0] - Start[0]) / 3)):
        draw.point([pixel, Start[1]], (0, 0, 0))
    for pixel in range(Start[0] + 2*int((End[0] - Start[0]) / 3), End[0]):
        draw.point([pixel, Start[1]], (0, 0, 0))
    for iterate in range(NumberOfIterates):
        LastColour = (0, 0, 0)
        ItvlStart = Start[0]
        for pixel in range(Start[0], End[0]+1):
            CurrentColour = im.getpixel((pixel, Start[1]))
            if LastColour != CurrentColour and LastColour == (0, 0, 0):
                ItvlLength = (pixel - 1) - ItvlStart
                for ItvlPixel in range(int(ItvlLength / 3)):
                    draw.point([ItvlStart + int(ItvlLength / 3) + ItvlPixel, Start[1]], (255, 255, 255))
            if LastColour != CurrentColour and LastColour == (255, 255, 255):
                ItvlStart = pixel - 1
            LastColour = CurrentColour
    im.save(filename + '.png', 'PNG')

#generate_cantor_png(5, [-0.1, 1.1, 0.05, -0.05], 2000, 'Cantor')

def topologists_sine_curve(NumberOfPointsCalculated, scale):
    
    # Generates an image of the Topologist's sine curve
    #
    # Inputs:
    # NumberOfPointsCalculated = integer that specifies how many points of sin(1/x) we calculate - a higher number takes more time but gives a more accurate depiction
    # scale = 4-tuple, specifying a rectangle of the real plane - order goes [left, right, top, bottom]
    #
    # Outputs:
    # pyplot will splash an image on-screen

    plt.axhline(0)
    plt.axvline(0)
    x = np.linspace(0.0001*scale[1], min(scale[1], 1), NumberOfPointsCalculated)
    plt.plot(x, np.sin(1/x), color = "red")
    plt.xlim(scale[0], scale[1])
    plt.ylim(scale[3], scale[2])
    plt.show()

#topologists_sine_curve(100000, [-0.001, 0.01, 0.1, -0.1])

def parameter_potential_function(c, n):

    # Calculates (an approximation to) the parameter potential function for the Mandelbrot set
    #
    # Inputs:
    # c = complex number that is the point in the parameter plane whose potential we are calculating
    # n = integer that we use to approximate the potential with; a higher n-value gives a more accurate answer,
    #     but even for, say, n > 10 we may sometimes get an error, as calculations involve high numbers - this
    #     isn't likely to be a problem though, as the approximation is good even for e.g. n = 10
    #
    # Outputs:
    # returns the approximation to the parameter potential of the Mandelbrot set at c
    
    arg = abs(iterate_f(c, 0, n))
    if arg < 1:
        arg = 0
    else:
        arg = log(arg)
    return arg / (2 ** n)

def parameter_equipotential(constant, thickness, EquiNumber, NumberOfIterates, scale, ImageWidth, filename):

    # Generates an image of the Mandelbrot set, enveloped by some equipotential
    #
    # Inputs:
    # constant = number that is the potential / radius of the equipotential we are drawing
    # thickness = number that describes the requested thickness of the drawn equipotential curve; higher number gives a thicker
    #             equipotential, I would recommend putting thickness = 0.001 or thereabouts, if the entire Mandelbrot set is in picture
    # EquiNumber = integer that will be used to approximate the potential function; a higher number gives a more accurate approximation,
    #              but any value of EquiNumber more than, say, 10 will return an error as calculations involve huge numbers
    # NumberOfIterates = integer that specifies how many iterates we calculate - a higher number takes more time but gives a more accurate depiction of Mandelbrot set
    # scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, top, bottom]
    # ImageWidth = integer that gives the desired number of pixels that the image has across - height will be calculated accordingly, based on scale
    # filename = string that the image of the equipotential will be saved as - note, this should not include the file extension, e.g. .png
    #
    # Outputs:
    # saves a .png image in the current working directory, depicting the equipotential
    
    generate_mandelbrot_png(NumberOfIterates, scale, ImageWidth, "auxiliary-im")
    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.open("auxiliary-im.png")
    draw = ImageDraw.Draw(im)
    for height in range(ImageHeight, 0, -1):
        for width in range(ImageWidth):
            num = convert_img_coord_to_complex(width, height, scale, ImageWidth, ImageHeight)
            difference = abs(constant - parameter_potential_function(num, EquiNumber))
            if difference < thickness:
                draw.point([width, height], (0, 0, 0))
    im.save(filename + '.png', 'PNG')
    
#parameter_equipotential(0.01, 0.001, 10, 100, [-2.5, 1, 1.75, -1.75], 800, "ParameterEquipotential")
