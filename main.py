# imports
import cv2 as cv
import numpy as np
from helpers import butter_bandpass_filter
from helpers import detect_face
from helpers import find_corners
from helpers import window
from helpers import interpolate_points
# from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA

interpolated = None

# Load Video
cap = cv.VideoCapture('test.mp4')
fps = cap.get(cv.CAP_PROP_FPS)

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

face_cascade = cv.CascadeClassifier()
if not face_cascade.load('./haarcascade_frontalface_alt.xml'):
    print('--(!)Error loading face cascade')
    exit(0)


feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

                       # Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# First frame
ret, prev_frame = cap.read()
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
# find corners
p0 = find_corners(prev_gray, face_cascade, feature_params)
firstp = p0

vertical_component = np.array([])
while(cap.isOpened()):
    ret, current_frame = cap.read()
    if ret == False:
            break
    current_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    # apply lucas Kanade optical flow for trajectories
    p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]
    good_first = firstp[st==1]

    vertical_component = np.append(vertical_component, ((good_new - good_first)[:, 1]))
    # changing frames
    prev_gray = current_gray.copy()
    p0 = p1.reshape(-1, 1, 2)

for data in window(vertical_component):
    points = np.array([p for p in data])
    # Interpolate points to 250 Hz
    try:
        interpolated = interpolate_points(np.vstack(points), fps=fps).T
    except ValueError:
        continue

# filtering data
filtered = butter_bandpass_filter(interpolated).T

# PCA decomposition
# First we remove the time-frames with the top 25%
norms = np.linalg.norm(filtered, 2, axis=1)
removed_abnormalities = filtered[norms > np.percentile(norms, 75)]
import ipdb; ipdb.set_trace()
pca = PCA()
pca.fit(removed_abnormalities)
transformed = pca.transform(filtered)

# Signal Selection 


# Finding bpm




# end script
cap.release()
cv.destroyAllWindows()
