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
    h = int(height * 0.05)
    # taking 25% of w
    w = int(width * 0.25)

    feature = frame[h:height-h, w:width-w, :]
    # removing eyes and similar features from the frame
    feature_height = feature.shape[0]
    feature_wo_eyes = feature[0:int(feature_height*0.2):int(feature_height*0.6), :, :]

    return feature_wo_eyes


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs=250, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y