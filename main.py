# imports
import cv as cv
import numpy as np
from helpers import butter_bandpass_filter
from helpers import detect_face
from helpers import feature_extraction
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


# Load Video
cap = cv.VideoCapture('somevideo.mp4')

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
prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
# detect face frame
face_frame = detect_face(prev_frame, face_cascade)
# Region Selection
roi_features_gray = cv.cvtColor(feature_extraction(face_frame), cv.COLOR_BGR2GRAY)
# Find Corners
p0 = cv.goodFeaturesToTrack(roi_features_gray, mask = None, **feature_params)


vertical_component = np.array([[]])
while(cap.isOpened()):
    ret, current_frame = cap.read()
    if ret == False:
            break
    current_frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    # apply lucas Kanade optical flow for trajectories
    p1, st, err = cv.calcOpticalFlowPyrLK(prev_frame_gray, current_frame_gray, p0, None, **lk_params)
    # we take the vertical component
    # TODO: check the output for x and y
    vertical_component.append(p1[0][1])

    # changing frames
    prev_frame_gray = current_frame_gray.copy()
    p0 = p1

# take x as timestamps and y as vertical components
x = None
y = None
# apply cubic spline interpolation
resampled_signal = interp1d(x, y, kind='cubic')

# butterworth filtering
filtered_signal = butter_bandpass_filter(resampled_signal, 0.75, 5)

# PCA decomposition
pca = PCA()
signals_list = pca.fit(filtered_signal)

# Signal Selection 


# Finding bpm




# end script
cap.release()
cv.destroyAllWindows()
