import cv2
import rects
import utils


class Face(object):
    """Define on facial features: face, mouth, nose and eyes."""

    def __init__(self):
        self.faceRect = None
        self.leftEyeRect = None
        self.rightEyeRect = None
        self.noseRect = None
        self.mouthRect = None


class FaceTracker(object):
    """A tracker for facial features: face, eyes, nose, mouth."""

    def __init__(self, scaleFactor=1.2, minNeighbors=2,
                 flags=cv2.CASCADE_SCALE_IMAGE):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags

        self._faces = []
        self._faceClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_frontalface_alt.xml')
        self._eyeClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_eye.xml')
        self._noseClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_nose.xml')
        self._mouthClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_mouth.xml')

    @property
    def faces(self):
        """The tracked facial features"""
        return self._faces

    def _detectOneObject(self, classifier, image, rect,
                         imageSizeToMinSizeRatio):
        x, y, w, h = rect

        minSize = utils.widthHeightDividedBy(
            image, imageSizeToMinSizeRatio)

        subImage = image[int(y):int(y + h), int(x):int(x + w)]

        subRects = classifier.detectMultiScale(subImage,
                                               self.scaleFactor,
                                               self.minNeighbors,
                                               self.flags, minSize)
        if len(subRects) == 0:
            return None

        subX, subY, subW, subH = subRects[0]
        return (x + subX, y + subY, subW, subH)

    def drawDebugRects(self, image):
        """Draw rectangles around the tracked facial features."""

        if utils.isGray(image):
            faceColor = 255
            leftEyeColor = 255
            rightEyeColor = 255
            noseColor = 255
            mouthColor = 255
        else:
            faceColor = (255, 255, 255)
            leftEyeColor = (0, 0, 255)
            rightEyeColor = (0, 255, 255)
            noseColor = (0, 255, 0)
            mouthColor = (255, 0, 0)

        for face in self.faces:
            rects.outlineRect(image, face.faceRect, faceColor)
            rects.outlineRect(image, face.leftEyeRect, leftEyeColor)
            rects.outlineRect(image, face.rightEyeRect, rightEyeColor)
            rects.outlineRect(image, face.noseRect, noseColor)
            rects.outlineRect(image, face.mouthRect, mouthColor)

    def update(self, image):
        """Update the tracked facial features."""

        self._faces = []

        if utils.isGray(image):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image, image)

        minSize = utils.widthHeightDividedBy(image, 8)

        faceRects = self._faceClassifier.detectMultiScale(image,
                                                          self.scaleFactor,
                                                          self.minNeighbors,
                                                          self.flags,
                                                          minSize)
        if faceRects is not None:
            for faceRect in faceRects:

                face = Face()
                face.faceRect = faceRect

                x, y, w, h = [int(i) for i in faceRect]

                # Seek an eye in the upper-left part of the face.
                searchRect = (x + w / 7, y, w * 2 / 7, h / 2)
                face.leftEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64)

                # Seek an eye in the upper-right part of the face.
                searchRect = (x + w * 4 / 7, y, w * 2 / 7, h / 2)
                face.rightEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64)

                # Seek an eye in the middle part of the face.
                searchRect = (x + w / 4, y + h / 4, w / 2, h / 2)
                face.noseEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64)

                # Seek an eye in the lower-middle part of the face.
                searchRect = (
                    x + w / 6,
                    y + h * 2 / 3,
                    w * 2 / 3,
                    h / 3)
                face.mouthEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64)

                self._faces.append(face)
