import cv2
import numpy as np
import utils


def outlineRect(image, rect, color):
    """used to draw a rectangle"""
    if rect is None:
        return
    X, y, w, h = [int(i) for i in rect]
    cv2.rectangle(image, (X, y), (X + w, y + h), color)


def copyRect(src, dst, srcRect, dstRect, mask=None,
             interpolation=cv2.INTER_LINEAR):
    """Copy part of the source to part of the destination"""
    x0, y0, w0, h0 = [int(i) for i in srcRect]
    x1, y1, w1, h1 = [int(j) for j in dstRect]

    # Resize the contents of the source sub-rectangle
    # Put the result in the destination subrectangle
    if mask is None:
        dst[y1:y1 + h1, x1:x1 + w1] = cv2.resize(src[y0:y0 + h0, x0:x0 + w0], (w1, h1),
                                                 interpolation=interpolation)
    else:
        if not utils.isGray(src):
            # Convert the mask to 3 channels, like the image.
            mask = mask.repeat(3).reshape(h0, w0, 3)
        # Perform the copy, with the mask applied.
        dst[y1:y1 + h1, x1:x1 + w1] = np.where(cv2.resize(mask,
                                                          (w1, h1),
                                                          interpolation=cv2.INTER_LINEAR),
                                               cv2.resize(src[y0:y0 + h0, x0:x0 + w0], (w1, h1),
                                                          interpolation=interpolation),
                                               dst[y1:y1 + h1, x1:x1 + w1])


def swapRects(src, dst, rects, masks=None,
              interpolation=cv2.INTER_LINEAR):
    """Copy the source with two or more sub-rectangles swapped."""
    if dst is not src:
        dst[:] = src
    numRects = len(rects)

    if numRects < 2:
        return

    if masks is None:
        masks = [None] * numRects

    # Copy the contents of last rectangle into temporary storage.
    x, y, w, h = rects[numRects - 1]
    temp = src[y:y + h, x:x + w].copy()

    # Copy the contents of each rectangle into next
    i = numRects - 2
    while i >= 0:
        copyRect(src, dst, rects[i], rects[i + 1], masks[i],
                 interpolation)
        i -= 1

    # Copy the temporarily stored content into the first rectangle
    copyRect(temp, dst, (0, 0, w, h), rects[0], masks[numRects - 1],
             interpolation)
