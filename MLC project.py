"""MLC project python code by Elliott Cawtheray, last updated 22nd April 2021
Written for my undergraduate final-year project.

It is hoped that the comments attached to code are sufficient to explain
how to use this software. If anything is unclear or you find any bugs,
I would happily be contacted at ec226.mathematics@gmail.com

There are commented-out example calls to functions, which I hope
are interesting examples to help you use this code!

Some of the code below was used for a specific purpose in my project
and was included pretty much only to demonstrate that the resulting
images were my own creation. So, don't be surprised if some functions
are more useful to you than others!

Feel free to use this code as you wish, and modify it as you wish :)
"""

import cmath
from matplotlib import pyplot as plt
from imageio import imread
import numpy as np
from PIL import Image, ImageDraw
import random
from math import floor, log

# In this module, i is reserved for the square root of -1
# Complex numbers are thus written as a + b*i, for some numbers a, b
i = cmath.sqrt(-1)

def f(c, z):
    """Calculates the value f_c(z)

    Inputs:
    c = complex number that is a parameter that defines the quadratic f_c
    z = complex number that is the input to the function

    Outputs:
    Complex number whose value is f_c(z) = z^2 + c
    """
    return z**2 + c

def iterate_f(c, z, n):
    """Calculates the nth iterate of f_c

    Inputs:
    c = complex number that is a parameter that defines the quadratic f_c
    z = complex number that is the initial input to the function
    n = integer specifying how many times to iterate

    Outputs:
    Complex number whose value is f_c^{\circ n} (z)
    """
    zNew = z
    for iterate in range(n):
        zNew = f(c, zNew)
    return zNew

def plot_orbit_f(c, z, NumberOfIterates, scale, DrawCircle="False"):
    """Plots the orbit of f_c on the dynamical plane

    Inputs:
    c = complex number, so that we are iterating z -> z^2 + c.
    z = complex number that is the initial point of the iteration
    NumberOfIterates = integer number of iterates to plot
    scale = 4-tuple, say (a, b, c, d), so that the plot shows the plane for a < Re z < b, c < Im z < d
    DrawCircle = boolean, if True then the circle C(0, 2) will also be shown

    Outputs:
    pyplot will splash an image on-screen
    """
    RealPart = []
    ImagPart = []
    point = z
    for iterate in range(NumberOfIterates):
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

#plot_orbit_f(0.25+ 0.1*i, 0, 100, [0.2, 0.35, 0.075, 0.275], False)
"""The above is a slight perturbation of the cusp parameter c = 0.25,
and so belongs to the interior of (the main cardioid of) M. Here, we
see the iterates in the critical orbit being attracted to a fixed point"""

def generate_hyperbolic_components_png(NumberOfComponents, scale, ImageWidth, filename):
    """Generates an image of hyperbolic components of the Mandelbrot set on the parameter plane, that is, connected components of the interior of the
    Mandelbrot set which consist of parameters whose quadratics possess attracting cycles
    
    Inputs:
    NumberOfComponents = integer specifying how many hyperbolic components to generate.
                         This function is work in progress - can currently only go up to 2 components
    scale = 4-tuple specifying which part of the complex plane we want to look at, so that we may e.g. zoom in - order goes [left, right, bottom, top]
    ImageWidth = integer that is number of pixels that the output image will have as width
    filename = string that the image will have as its name - file extensions need not be included in this
    
    Outputs:
    Will save a .png image in the current working directory
    """
    scale = [scale[0], scale[1], scale[3], scale[2]]
    if NumberOfComponents not in [1, 2]:
        raise ValueError("NumberOfComponents must be 1 or 2.")
    ratio = abs(scale[2] - scale[3]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x in range(0, ImageWidth):
        for y in range(0, ImageHeight):
            c = scale[0] + (x / ImageWidth) * (scale[1] - scale[0]) + (scale[2] + (y / ImageHeight) * (scale[3] - scale[2])) * i
            if abs(1-cmath.sqrt(1-4*c)) < 1:
                draw.point([x, y], (0, 0, 0))
            elif abs(4*(c + 1)) < 1:
                if NumberOfComponents > 1:
                    draw.point([x, y], (127, 127, 127))
            else:
                draw.point([x, y], (255, 255, 255))
    im.save(filename + '.png', 'PNG')
    
#generate_hyperbolic_components_png(2, [-1.5, 0.5, 1, -1], 1000, "2components")
"""Shows the main cardioid and the period-2 hyperbolic component"""

def plot_image_on_plane(PlaneScale, ImageScale, filename, *args):
    """Plots an image on the complex plane, with added axes on border

    Inputs:
    PlaneScale = 4-tuple specifying the part of the complex plane that we map image on to - order goes [left, right, top, bottom]
    ImageScale = 4-tuple specifying where image goes on plane - order goes [left, right, top, bottom]
    filename = string that is the filename (including extension) of the image, which must be in the current working directory
    *args = optional input, if you want to specify the numbers on the axes, you can write 6 additional inputs, and we will have x-axis numbers
            from args[0] to args[1] in steps of length args[2], and y-axis numbers from args[3] to args[4] in steps of length args[5]

    Outputs:
    pyplot will splash an image on-screen
    """
    PlaneScale = [PlaneScale[0], PlaneScale[1], PlaneScale[3], PlaneScale[2]]
    ImageScale = [ImageScale[0], ImageScale[1], ImageScale[3], ImageScale[2]]
    try:
        img = imread(filename)
    except FileNotFoundError:
        raise FileNotFoundError("file '" + filename + "' not found in current working directory for use in plot_image_on_plane. Did you include the file extension?")
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

#plot_image_on_plane([-1.5, 0.5, 1, -1], [-1.5, 0.5, 1, -1], '2components.png')
"""Note: will only work if you have generated the 2components.png image from the previous function.
Shows the main cardioid and period-2 hyperbolic component, plotted on the complex plane"""

def convert_complex_to_img_coord(z, scale, ImageWidth, ImageHeight):
    """Takes a complex number and calculates its coordinate as a PIL image, according to the part of the complex plane pictured (scale),
    and the number of pixels in the image (ImageWidth, ImageHeight)

    Inputs:
    z = complex number whose image coordinates we will calculate
    scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, top, bottom]
    ImageWidth = number of pixels that the image has for its width
    ImageHeight = number of pixels that the image has for its height

    Outputs:
    returns a 2-tuple that gives us image coordinates. According to PIL's logic, the 1st entry is how far along we are from the left,
    and 2nd is how far down from the top
    """
    HowFarAlong = (z.real - scale[0]) / abs(scale[1] - scale[0])
    HowFarDown = (scale[2] - z.imag) / abs(scale[3] - scale[2])
    return (floor(HowFarAlong * ImageWidth), floor(HowFarDown * ImageHeight))

def convert_img_coord_to_complex(x, y, scale, ImageWidth, ImageHeight):
    """Converts PIL image coordinates to a complex number, according to which part of the complex plane we are on and the size of the image

    Inputs:
    x = integer that says how many pixels we are from the left of the image
    y = integer that says how many pixels we are from the top of the image
    scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, top, bottom]
    ImageWidth = number of pixels that the image has for its width
    ImageHeight = number of pixels that the image has for its height

    Outputs:
    returns a complex number, corresponding to the image coordinates in the plane
    """
    HowFarAlong = x / ImageWidth
    RealPart = scale[0] + HowFarAlong * abs(scale[1] - scale[0])
    HowFarDown = y / ImageHeight
    ImagPart = scale[2] - HowFarDown * abs(scale[3] - scale[2])
    return RealPart + ImagPart*i

def generate_mandelbrot_png(NumberOfIterates, scale, ImageWidth, filename):
    """Generates an image of the Mandelbrot set on the parameter plane

    Inputs:
    NumberOfIterates = integer that is max number of iterates to which we test whether a point is in the set.
                       A higher value will give a more accurate image, but will take longer
    scale = 4-tuple specifying which part of the complex plane we want to look at, so that we may e.g. zoom in - order goes [left, right, top, bottom]
    ImageWidth = integer that is number of pixels that the output image will have as width
    filename = string that the image will have as its name - file extensions need not be included in this

    Outputs:
    Will save a .png image in the current working directory, displaying the specified part of the Mandelbrot set
    """
    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x in range(0, ImageWidth):
        for y in range(0, ImageHeight):
            c = convert_img_coord_to_complex(x, y, scale, ImageWidth, ImageHeight)
            n = 0
            z = 0
            while abs(z) <= 2 and n < NumberOfIterates:
                n = n + 1
                z = f(c, z)
            ColourNum = 255 - int(n * 255 / NumberOfIterates)
            draw.point([x, y], (ColourNum, ColourNum, ColourNum))
    im.save(filename + '.png', 'PNG')

#generate_mandelbrot_png(200, [-0.18, -0.14, 1.05, 1.02], 500, "Minibrot")
"""Small copy of the Mandelbrot set near c = -0.16 + 1.035i"""

def generate_colour_mandelbrot_png(NumberOfIterates, scale, ImageWidth, ColourScheme):
    """Generates a colourful image of the Mandelbrot set on the parameter plane

    Inputs:
    NumberOfIterates = integer that is max number of iterates to which we test whether a point is in the set
                       A higher value will give a more accurate image, but will take longer
    scale = 4-tuple specifying which part of the complex plane we want to look at, so that we may e.g. zoom in - order goes [left, right, top, bottom]
    ImageWidth = integer that is number of pixels that the output image will have as width
    ColourScheme = string specifying the colour map to be used - google "matplotlib cmaps" for a list of the different colourmaps

    Outputs:
    pyplot will splash an image on-screen
    """
    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    img = np.full((ImageHeight, ImageWidth), 255)
    for x in range(0, ImageWidth):
        for y in range(0, ImageHeight):
            c = convert_img_coord_to_complex(x, y, scale, ImageWidth, ImageHeight)
            n = 0
            z = 0
            while abs(z) <= 2 and n < NumberOfIterates:
                n = n + 1
                z = f(c, z)
            ColourNum = 255 - int(n * 255 / NumberOfIterates)
            img[y][x] = ColourNum
    plt.imshow(img, cmap=ColourScheme)
    plt.axis("off")
    plt.show()

#generate_colour_mandelbrot_png(50, [-2.1, 0.6, 1.2, -1.2], 600, "nipy_spectral")
"""The entire Mandelbrot set, in glorious technicolour"""

def generate_julia_png(c, NumberOfIterates, scale, ImageWidth, filename):
    """Generates an image of the Julia set of a point in the parameter plane
    NOTE: In contrast to alt_generate_julia_png, this function uses the inverse iteration algorithm
    Accordingly, it is more efficient than the other version, but gives less detail for certain Julia sets

    Inputs:
    c = complex number that is a point in the parameter plane, so that we are considering the Julia set of f_c
    NumberOfIterates = integer that specifies how many iterates we calculate
                       A higher number takes more time but gives a more accurate depiction
    scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, bottom, top]
    ImageWidth = integer that gives the desired number of pixels that the image has across - height will be calculated accordingly, based on scale
    filename = string that the image of the Julia set will be saved as - note, this should not include the file extension

    Outputs:
    Will save a .png image in the current working directory, depicting the Julia set of f_c
    """
    scale = [scale[0], scale[1], scale[3], scale[2]]
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

#generate_julia_png(i, 500000, [-1.5, 1.5, -1.5, 1.5], 500, "Dendrite")
"""The Julia set J(i), which is a dendrite since c = i is a Misiurewicz parameter"""

def alt_generate_julia_png(c, NumberOfIterates, scale, ImageWidth, filename):
    """Generates an image of the Julia set of a point in the parameter plane
    NOTE: In contrast to generate_julia_png, this function uses the so-called boundary scanning method
    Accordingly, it is less efficient than the other version, but gives (far) more detail for certain Julia sets

    Inputs:
    c = complex number that is a point in the parameter plane, so that we are considering the Julia set of f_c
    NumberOfIterates = integer that specifies how many iterates we calculate
                       A higher number takes more time but gives a more accurate depiction
    scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, top, bottom]
    ImageWidth = integer that gives the desired number of pixels that the image has across - height will be calculated accordingly, based on scale
    filename = string that the image of the Julia set will be saved as - note, this should not include the file extension

    Outputs:
    saves a .png image in the current working directory, depicting the Julia set of f_c
    """
    scale = [scale[0], scale[1], scale[3], scale[2]]
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

#alt_generate_julia_png(-0.12+0.75*i, 500, [-1.5, 1.5, -1.5, 1.5], 400, "Rabbit")
"""The Douady Rabbit Julia set J(c), where f_c has a super-attracting 3-cycle"""

def generate_colour_julia_png(c, NumberOfIterates, scale, ImageWidth, ColourScheme):
    """Generates a colourful image of the Julia set of a point in the parameter plane
    Uses simple forward iteration method - not the quickest method.

    Inputs:
    c = complex number that is a point in the parameter plane, so that we are considering the Julia set of f_c
    NumberOfIterates = integer that specifies how many iterates we calculate - a higher number takes more time but gives a more accurate depiction
    scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, top, bottom]
    ImageWidth = integer that gives the desired number of pixels that the image has across - height will be calculated accordingly, based on scale
    ColourScheme = string specifying the colour map to be used - google "matplotlib cmaps" for a list of the different colourmaps

    Outputs:
    pyplot will splash an image on-screen
    """
    scale = [scale[0], scale[1], scale[3], scale[2]]
    ratio = abs(scale[2] - scale[3]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    img = np.full((ImageHeight, ImageWidth), 255)
    for x in range(0, ImageWidth):
        for y in range(0, ImageHeight):
            z = convert_img_coord_to_complex(x, y, scale, ImageWidth, ImageHeight)
            n = 0
            while abs(z) <= 2 and n < NumberOfIterates:
                n = n + 1
                z = f(c, z)
            ColourNum = 255 - int(n * 255 / NumberOfIterates)
            img[y][x] = ColourNum
    plt.imshow(img, cmap=ColourScheme)
    plt.axis("off")
    plt.show()

#generate_colour_julia_png(-0.12+0.75*i, 200, [-1.5, 1.5, -1.5, 1.5], 500, "gist_stern_r")
"""The Douady Rabbit displayed with the 'gist_stern_r' colourmap"""


"""The below generates a picture of dynamical rays of the Douady Rabbit
using explicit coordinates for the rays found manually - unlikely to be useful to anyone else!

def divide_piecewise_curve(ListOfPoints, NumberOfPointsToDivideInto):
    DividedList = [ListOfPoints[0]]
    NumberOfItvls = len(ListOfPoints) - 1
    NumberOfPointsForEachPiece = floor(NumberOfPointsToDivideInto / NumberOfItvls)
    for itvl in range(NumberOfItvls):
        a = ListOfPoints[itvl].real
        b = ListOfPoints[itvl].imag
        c = ListOfPoints[itvl + 1].real
        d = ListOfPoints[itvl + 1].imag
        gradient = (b - d) / (a - c)
        intercept = b - gradient*a
        step = (c - a) / NumberOfPointsForEachPiece
        CurrentX = a
        for divider in range(NumberOfPointsForEachPiece):
            CurrentX += step
            y = gradient*CurrentX + intercept
            DividedList.append(CurrentX + y*i)
        DividedList.append(ListOfPoints[itvl + 1])
    return DividedList

RabbitRay1 = [-0.27219+0.48036*i, -0.341+0.631*i, -0.365+0.726*i, -0.389+0.773*i, -0.418+0.873*i, -0.453+1.014*i, -0.542+1.491*i]
RabbitRay1 = divide_piecewise_curve(RabbitRay1, 10000)

RabbitRay2 = [-0.27219+0.48036*i, -0.165+0.49*i, -0.071+0.508*i, 0.023+0.543*i, 0.117+0.573*i, 0.206+0.626*i, 0.294+0.679*i, 0.394+0.767*i, 0.912+1.25*i, 1+1.332*i, 1.153+1.5*i]
RemovedBit = [0.488+0.826*i, 0.6+0.92*i, 0.7+1.014*i, 0.8+1.126*i]
RabbitRay2 = divide_piecewise_curve(RabbitRay2, 10000)

def manual_ray_draw(c, NumberOfIterates, scale, ImageWidth, ListOfListsOfPointsInRay, LineThickness, filename):
    alt_generate_julia_png(c, NumberOfIterates, [scale[0], scale[1], scale[3], scale[2]], ImageWidth, "auxiliary-im")
    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.open("auxiliary-im.png")
    draw = ImageDraw.Draw(im)
    for ListOfPointsInRay in ListOfListsOfPointsInRay:
        for indexer in range(len(ListOfPointsInRay) - 1):
            LastPoint = ListOfPointsInRay[indexer]
            NextPoint = ListOfPointsInRay[indexer + 1]
            if abs(LastPoint - NextPoint) < 0.1:
                Draw1 = convert_complex_to_img_coord(NextPoint, scale, ImageWidth, ImageHeight)
                Draw2 = convert_complex_to_img_coord(LastPoint, scale, ImageWidth, ImageHeight)
                draw.line((Draw1[0], Draw1[1], Draw2[0], Draw2[1]), fill=128, width=LineThickness)
    im.save(filename + '.png', 'PNG')

manual_ray_draw(-0.12+0.75*i, 400, [-1.5, 1.5, 1.5, -1.5], 300, [RabbitRay2], 2, "RabbRay2comp2")

def PreImagesOfRay(c, Ray):
    PreImageRay1 = []
    for RayCoord in Ray:
        PreIm = cmath.sqrt(RayCoord - c)
        PreImageRay1.append(PreIm)
    PreImageRay2 = []
    for RayCoord in Ray:
        PreIm = -1 * cmath.sqrt(RayCoord - c)
        PreImageRay2.append(PreIm)
    return [PreImageRay1, PreImageRay2]

PreImage1 = PreImagesOfRay(-0.12+0.75*i, RabbitRay2)[0]
PreImage1_1 = PreImagesOfRay(-0.12+0.75*i, PreImage1)[0]
PreImage1_2 = PreImagesOfRay(-0.12+0.75*i, PreImage1)[1]

PreImage2 = PreImagesOfRay(-0.12+0.75*i, RabbitRay2)[1]
PreImage2_1 = PreImagesOfRay(-0.12+0.75*i, PreImage2)[0]
PreImage2_2 = PreImagesOfRay(-0.12+0.75*i, PreImage2)[1]

PreImage3 = PreImagesOfRay(-0.12+0.75*i, RabbitRay1)[0]
PreImage3_1 = PreImagesOfRay(-0.12+0.75*i, PreImage3)[0]
PreImage3_2 = PreImagesOfRay(-0.12+0.75*i, PreImage3)[1]

PreImage4 = PreImagesOfRay(-0.12+0.75*i, RabbitRay1)[1]
PreImage4_1 = PreImagesOfRay(-0.12+0.75*i, PreImage4)[0]
PreImage4_2 = PreImagesOfRay(-0.12+0.75*i, PreImage4)[1]

# PreImage3 and PreImage4 are somehow mixed up
PreImage3redux = [PreImage3[0]]
for entry in range(len(PreImage3) - 1):
    try:
        if abs(PreImage3[entry + 1] - PreImage3redux[entry]) < abs(PreImage4[entry + 1] - PreImage3redux[entry]):
            PreImage3redux.append(PreImage3[entry + 1])
        else:
            PreImage3redux.append(PreImage4[entry + 1])
    except IndexError:
        pass
PreImage3 = PreImage3redux

manual_ray_draw(-0.12+0.75*i, 1000, [-1.5, 1.5, 1.5, -1.5], 500, [RabbitRay2, RabbitRay1,
                                                                 PreImage2], 2, "RabbPuzzleRays0")


manual_ray_draw(-0.12+0.75*i, 1000, [-1.5, 1.5, 1.5, -1.5], 500, [RabbitRay2, RabbitRay1,
                                                                 PreImage1,
                                                                 PreImage2, PreImage3, PreImage2_1], 2, "RabbPuzzleRays1")
"""

def plot_orbit_on_julia(c, z, NumberOfIteratesInJulia, NumberOfIteratesInOrbit, scale, ImageWidth, RenormNumber=1):
    """Plots the orbit of f_c on the dynamical plane

    Inputs:
    c = complex number, so that we are iterating f_c
    z = complex number that is initial point of iteration
    NumberOfIteratesInJulia = integer which decides 'resolution' of Julia set
    NumberOfIteratesInOrbit = integer number of points in orbit to plot
    scale = 4-tuple, say (a, b, c, d), so that the plot shows the plane for a < Re z < b, c < Im z < d
    ImageWidth = integer that gives number of pixels of the width of the generated image
    RenormNumber = Defaults to 1. Other inputs allow the plotted orbit to skip over
                   a fixed number of iterates, e.g. if RenormNumber = 3, we plot every 3rd point in orbit

    Outputs:
    pyplot will splash an image on-screen
    """
    alt_generate_julia_png(c, NumberOfIteratesInJulia, scale, ImageWidth, "auxiliary-im")
    scale = [scale[0], scale[1], scale[3], scale[2]]
    img = imread("auxiliary-im.png")
    RealPart = []
    ImagPart = []
    point = z
    WhichIterate = 0
    for iterate in range(NumberOfIteratesInOrbit):
        if WhichIterate % RenormNumber == 0:
            RealPart.append(point.real)
            ImagPart.append(point.imag)
        point = f(c, point)
        WhichIterate = WhichIterate + 1
    plt.plot(RealPart, ImagPart, 'C3', lw=3)
    plt.scatter(RealPart, ImagPart, s=120)
    plt.xlim(scale[0], scale[1])
    plt.ylim(scale[3], scale[2])
    plt.xlabel('Re z')
    plt.ylabel('Im z')
    plt.axes().set_aspect(1)
    plt.imshow(img, zorder=0, extent=[scale[0], scale[1], scale[3], scale[2]])
    plt.show()

#plot_orbit_on_julia(-0.12+0.75*i, 0.2, 300, 10, [-1.5, 1.5, -1.5, 1.5], 400, 1)
"""Iterates of the Douady rabbit starting near 0 are attracted to the super-attracting cycle"""

def topologists_sine_curve(NumberOfPointsCalculated, scale):
    """Generates an image of the Topologist's sine curve

    Inputs:
    NumberOfPointsCalculated = integer that specifies how many points of sin(1/x) we calculate
                               A higher number takes more time but gives a more accurate depiction
    scale = 4-tuple, specifying a rectangle of the real plane - order goes [left, right, bottom, top]

    Outputs:
    pyplot will splash an image on-screen
    """
    plt.axhline(0)
    plt.axvline(0)
    x = np.linspace(0.0001*scale[1], min(scale[1], 1), NumberOfPointsCalculated)
    plt.plot(x, np.sin(1/x), color = "red")
    plt.xlim(scale[0], scale[1])
    plt.ylim(scale[2], scale[3])
    plt.show()

#topologists_sine_curve(100000, [-0.001, 0.01, -0.1, 0.1])
"""A small neighbourhood of 0, containing infinitely many connected components
and thus demonstrating that the topologist's sine curve is not locally connected"""

def bottcher_mapping1(c, z):
    boolean = True
    n = 1
    while boolean:
        try:
            value = iterate_f(c, z, n) / (2 ** n)
        except OverflowError:
            boolean = False
            return value
        n = n + 1

def bottcher_mapping2(c, z, n):
    return iterate_f(c, z, n) / (2 ** n)

def e_to_the_2_pi_i(theta):
    return np.cos(2 * np.pi * theta) + i * np.sin(2 * np.pi * theta)

def potential_function(c, z, n):
    """Calculates (an approximation to) the potential function G_c(z) of the Julia set K(c), evaluated at z
    Note that taking z = c will give the potential function for the Mandelbrot set, evaluated at the parameter c

    Inputs:
    c = complex number that is the parameter associated to the Julia set whose potential we are calculating
    z = complex number that is where we are evaluating G_c
    n = integer that we use to approximate the potential with; a higher n-value gives a more accurate answer,
        but even for, say, n > 10 we may sometimes get an error, as calculations involve high numbers - this
        isn't likely to be a problem though, as the approximation is good even for e.g. n = 10

    Outputs:
    Returns the value of G_c(z)
    """
    arg = abs(iterate_f(c, z, n))
    if arg < 1:
        arg = 0
    else:
        arg = log(arg)
    return arg / (2 ** n)

def dynamical_equipotential(c, potential, thickness, EquiNumber, NumberOfIterates, scale, ImageWidth, filename):
    """Generates an image of the Julia set J(c), enveloped by some equipotential

    Inputs:
    c = complex number, so that we are drawing J(c) and one of its equipotentials
    potential = float that is the potential / radius of the equipotential we are drawing
    thickness = float that describes the requested thickness of the drawn equipotential curve; higher number gives a thicker
                equipotential, I would recommend putting thickness as half of potential, seeing what you get, and then fiddling with it
    EquiNumber = integer that will be used to approximate the potential function; a higher number gives a more accurate approximation,
                 but any value of EquiNumber more than, say, 10 will return an error as calculations involve huge numbers
    NumberOfIterates = integer that specifies how many iterates we calculate
                       A higher number takes more time but gives a more accurate depiction of Julia set
    scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, bottom, top]
    ImageWidth = integer that gives the desired number of pixels that the image has across - height will be calculated accordingly, based on scale
    filename = string that the image of the equipotential will be saved as

    Outputs:
    Will save a .png image in the current working directory, depicting the equipotential
    """
    alt_generate_julia_png(c, NumberOfIterates, scale, ImageWidth, "auxiliary-im")
    scale = [scale[0], scale[1], scale[3], scale[2]]
    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.open("auxiliary-im.png")
    draw = ImageDraw.Draw(im)
    for height in range(ImageHeight, 0, -1):
        for width in range(ImageWidth):
            num = convert_img_coord_to_complex(width, height, scale, ImageWidth, ImageHeight)
            difference = abs(potential - potential_function(c, num, EquiNumber))
            if difference < thickness:
                draw.point([width, height], (0, 0, 0))
    im.save(filename + '.png', 'PNG')

#dynamical_equipotential(-0.12+0.75*i, 0.1, 0.005, 9, 250, [-1.5, 1.5, -1.5, 1.5], 500, "RabbitEquipotential")
"""The Douady Rabbit with an equipotential"""

def parameter_equipotential(potential, thickness, EquiNumber, NumberOfIterates, scale, ImageWidth, filename):
    """Generates an image of the Mandelbrot set, enveloped by some equipotential

    Inputs:
    potential = float that is the potential / radius of the equipotential we are drawing
    thickness = float that describes the requested thickness of the drawn equipotential curve; higher number gives a thicker
                equipotential, I would recommend putting thickness as half of potential, seeing what you get, and then fiddling with it
    EquiNumber = integer that will be used to approximate the potential function; a higher number gives a more accurate approximation,
                 but any value of EquiNumber more than, say, 10 will return an error as calculations involve huge numbers
    NumberOfIterates = integer that specifies how many iterates we calculate
                       A higher number takes more time but gives a more accurate depiction of Mandelbrot set
    scale = 4-tuple, specifying a rectangle of the complex plane - order goes [left, right, bottom, top]
    ImageWidth = integer that gives the desired number of pixels that the image has across - height will be calculated accordingly, based on scale
    filename = string that the image of the equipotential will be saved as

    Outputs:
    Will save a .png image in the current working directory, depicting the equipotential
    """
    generate_mandelbrot_png(NumberOfIterates, scale, ImageWidth, "auxiliary-im")
    ratio = abs(scale[3] - scale[2]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.open("auxiliary-im.png")
    draw = ImageDraw.Draw(im)
    for height in range(ImageHeight, 0, -1):
        for width in range(ImageWidth):
            num = convert_img_coord_to_complex(width, height, scale, ImageWidth, ImageHeight)
            difference = abs(potential - potential_function(num, num, EquiNumber))
            if difference < thickness:
                draw.point([width, height], (0, 0, 0))
    im.save(filename + '.png', 'PNG')

#parameter_equipotential(0.1, 0.005, 8, 50, [-2.2, 0.8, 1.3, -1.3], 500, "MandelbrotEquipotential")
"""The Mandelbrot set with some equipotential"""

def draw_period_doubling_bifurcation_components(scale, ImageWidth, filename):
    """Generates an image of the hyperbolic components of the Mandelbrot set along the real line
    other than the main cardioid and some of the smaller components nearest to Feigenbaum parameter

    Inputs:
    scale = 4-tuple specifying which part of the complex plane we want to look at, so that we may e.g. zoom in - order goes [left, right, bottom, top]
    ImageWidth = integer that is number of pixels that the output image will have as width
    filename = string that the image will have as its name - file extensions need not be included in this

    Outputs:
    Saves a .png image in the current working directory, displaying the hyperbolic components
    """
    scale = [scale[0], scale[1], scale[3], scale[2]]
    ratio = abs(scale[2] - scale[3]) / abs(scale[1] - scale[0])
    ImageHeight = int(ImageWidth * ratio)
    im = Image.new('RGB', (ImageWidth, ImageHeight), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    bif_param = [-0.75, -1.25, -1.3680989, -1.3940462, -1.3996312, -1.4008287, -1.4010853, -1.4011402, -1.401151982029, -1.401154502237]
    Circles = []
    for iterator in range(len(bif_param) - 1):
        radius = abs(bif_param[iterator] - bif_param[iterator + 1]) / 2
        centre = bif_param[iterator] - radius
        Circles.append((centre, radius))
    for x in range(0, ImageWidth):
        for y in range(0, ImageHeight):
            c = scale[0] + (x / ImageWidth) * (scale[1] - scale[0]) + (scale[2] + (y / ImageHeight) * (scale[3] - scale[2])) * i
            InACircle = False
            for circle in Circles:
                if abs(c - circle[0]) < circle[1]:
                    InACircle = True
                    draw.point([x, y], (0, 0, 0))
            if not InACircle:
                draw.point([x, y], (255, 255, 255))
    im.save(filename + '.png', 'PNG')

draw_period_doubling_bifurcation_components([-1.45, -0.7, 0.3, -0.3], 1000, "feigenbaum")
"""The circular hyperbolic components in the cascade of period doubling bifurcations
along the real line in M - the feigenbaum parameter lives at the left-most point of the circles"""
