import cv2 as cv
import numpy as np
import itertools
from scipy.signal import butter
from scipy.signal import lfilter
from scipy import interpolate
from scipy.interpolate import CubicSpline


def window(seq, n=2, skip=1):
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    i = 0
    for elem in it:
        result = result[1:] + (elem,)
        i = (i + 1) % skip
        if i == 0:
            yield result


def detect_face(frame, cascade):
    frame_gray = cv.equalizeHist(frame)
    #-- Detect faces
    faces = cascade.detectMultiScale(frame_gray, 1.3, 5)
    return faces


def make_face_rects(rect):

    x, y, w, h = rect

    rect1_x = x + w / 4.0
    rect1_w = w / 2.0
    rect1_y = y + 0.05 * h
    rect1_h = h * 0.9 * 0.2

    rect2_x = rect1_x
    rect2_w = rect1_w

    rect2_y = y + 0.05 * h + (h * 0.9 * 0.55)
    rect2_h = h * 0.9 * 0.45

    return (
        (int(rect1_x), int(rect1_y), int(rect1_w), int(rect1_h)),
        (int(rect2_x), int(rect2_y), int(rect2_w), int(rect2_h))
    )


# def feature_extraction(frame):
#     height, width = frame.shape[0], frame.shape[1]
#     # taking 5% of height
#     hi = int(height * 0.05)
#     # taking 25% of w
#     wi = int(width * 0.25)

#     feature = frame[hi:height-hi, wi:width-wi, :]
#     # removing eyes and similar features from the frame
#     feature_height = feature.shape[0]
#     # TODO: Slice the middle subrectangle
#     # first_slice = feature[0:int(feature_height*0.2)]
#     # second_slice = feature[int(feature_height*0.6):feature.shape[0]]

#     feature_wo_eyes = feature[0:int(feature_height*0.3), :, :]

#     return feature_wo_eyes


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut=0.75, highcut=5, fs=250, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def find_corners(prev_gray, face_cascade, feature_params):
    # detect face frame
    rects = detect_face(prev_gray, face_cascade)
    # create mask and fill detected areas with white except eyes
    mask = np.zeros_like(prev_gray)
    for x, y, w, h in rects:
        # Fill in a rectangle area of the 'mask' array white
        cv.rectangle(mask, (x, y), ((x + w), (y + h)),
                        thickness=-1,
                        color=(255, 255, 255))
    # Find Corners
    p0 = cv.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)
    return p0


def interpolate_points(points, fps=30, sample_freq=250, axis=0):
    N = points.shape[axis]
    indices = np.arange(0, N)
    # Make an 'interpolation function' using scikit's interp1d
    f = CubicSpline(indices, points, axis=axis)
    # Define the new time axis,
    xnew = np.arange(0, N - 1, fps/sample_freq)
    return f(xnew)