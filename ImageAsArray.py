"""
    This is a function that opens a single image and do a series of data augmentation operations to generate a single sample

    It is inneficient to open a file each time we want to generate a sample

    But we must shuffle the examples which will go into the trainning set

    If we open an image file and try to do all possible transformations on the file, and at the end shuffle all the
trainning set, we could gain from open/close efficiency - but the resulting trainning set may not fit
on memory and/or disk
"""
# imports
import numpy as np
import PIL
from PIL import Image, ImageFilter
import Constants as const
import math

def GetImagePreprocessedArray(file, 
                            filter = None,
                            tone = 0,
                            stepH = 0,
                            stepW = 0,
                            degree = 0):
    """
    Arguments:
        file: image file path
        filter: filter to be applied to image
        tone: integer to be added to each color component (R, G and B)
        stepH: image displacement in height pixels
        stepW: image displacement in width pixels
        degree: rotation angle

    """

    # open image
    img = Image.open(file)

    # get image size
    w, h = img.size

    # apply filter
    if filter == None:
        filterImg = img
    else:
        filterImg = img.filter(filter)

    arr = np.array(filterImg).astype(np.int32)

    # apply tone (carefull about overflow)
    arr2 = arr + tone
    arr2[arr2 < 0] = 0
    arr2[arr2 > 255] = 255

    # displace on height and width
    if stepH < 0:
        sliceOrigH = slice(0, h+(stepH))
        sliceDestH = slice(-stepH, h)
    else:
        sliceOrigH = slice(stepH, h)
        sliceDestH = slice(0, h-stepH)

    if stepW < 0:
        sliceOrigW = slice(0,w+stepW)
        sliceDestW = slice(-stepW,w)
    else:
        sliceOrigW = slice(stepW,w)
        sliceDestW = slice(0,w-stepW)

    arr2[sliceDestH,sliceDestW,:] = arr2[sliceOrigH,sliceOrigW,:]

    # rotate
    if degree != 0:
        imgRot = Image.fromarray(arr2.astype(np.uint8)).convert('RGBA').rotate(degree)
        imgRotBack = Image.new("RGBA", imgRot.size, (255, 255, 255, 255))
        arr2 = np.array(Image.composite(imgRot, imgRotBack, imgRot)).astype(np.int32)

    # resize to match NN input size
    if (arr2.shape[0] != const.X_Height) or (arr2.shape[1] != const.X_Width):
        arr2 = np.array(Image.fromarray(arr2.astype(np.uint8)).resize(size=(const.X_Width, const.X_Height), resample=PIL.Image.ANTIALIAS))

    # apply mask to avoid noise outside of circle
    # important: as we're using images from fisheye cameras, the Region Of Interest is a circle
    masked_arr = arr2.copy()
    masked_arr[~createCircularMask(const.X_Width, const.X_Height)] = 255

    # return (np.expand_dims(masked_arr, axis=2)/np.float16(255.)).astype(np.float16)
    return (masked_arr/np.float16(255.)).astype(np.float16)

def createCircularMask(h, w, center=None, radius=None):
    """
    this is an auxiliary function to create a circular mask over the image
    remember that our images were obtained from fisheye lenses, so the ROI is a circle...
    """

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask