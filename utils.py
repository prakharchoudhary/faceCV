import cv2
import numpy as np
import scipy.interpolate


def createCurveFunc(points):
    """Return a function derived from control points."""
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None
    xs, ys = zip(*points)
    if numPoints < 4:
        kind = 'linear'
        # 'quadratic' is not implemented.
    else:
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs, ys, kind,
                                      bounds_error=False)


def createLookupArray(func, length=256):
    """Return a lookup for whole-number inputs to a function.

    The lookup values are clamped to [0, length - 1].
    """
    if func is None:
        return None
    lookupArray = np.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookupArray[i] = min(max(0, func_i), length - 1)
        i += 1
    return lookupArray


def applyLookupArray(lookupArray, src, dst):
    """Map a source to a destination using a loop."""
    if lookupArray is None:
        return
    dst[:] = lookupArray[src]


def createCompositeFunc(func0, func1):
    """Return a composite of two functions."""
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))


def createFlatView(array):
    """Return a 1D view of an array of any dimensionality."""
    flatView = array.view()
    flatView.shape = array.size
    return flatView


def isGray(image):
    """Return if the image has one channel per pixel."""
    return image.ndim < 3


def widthHeightDividedBy(image, divisor):
    """Return an image's dimensions, divided by a value."""
    h, w = image.shape[:2]
    return (int(w / divisor), int(h / divisor))
