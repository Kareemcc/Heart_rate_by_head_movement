import cv2 as cv
from scipy.signal import butter
from scipy.signal import lfilter


def detect_face(frame, cascade):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        face_roi = frame[y:y+h,x:x+w]
    return face_roi


def feature_extraction(frame):
    height, width = frame.shape[0], frame.shape[1]
    # taking 5% of height
    hi = int(height * 0.05)
    # taking 25% of w
    wi = int(width * 0.25)

    feature = frame[hi:height-hi, wi:width-wi, :]
    # removing eyes and similar features from the frame
    feature_height = feature.shape[0]
    # TODO: Slice the middle subrectangle
    # first_slice = feature[0:int(feature_height*0.2)]
    # second_slice = feature[int(feature_height*0.6):feature.shape[0]]

    feature_wo_eyes = feature[0:int(feature_height*0.3), :, :]

    return feature_wo_eyes


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs=250, order=5):
    import ipdb; ipdb.set_trace()
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


def find_corners(prev_frame, face_cascade, feature_params):
    # detect face frame
    face_frame = detect_face(prev_frame, face_cascade)
    # Region Selection
    roi_features_gray = cv.cvtColor((face_frame), cv.COLOR_BGR2GRAY)
    # Find Corners
    p0 = cv.goodFeaturesToTrack(roi_features_gray, mask = None, **feature_params)
    return p0