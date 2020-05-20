import cv2
import time

webcam = cv2.VideoCapture(0)
outputVideoFilename = "test.mp4"
outputVideoWriter = cv2.VideoWriter()
outputVideoWriter.open(outputVideoFilename, cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480), True)
t_end = time.time() + 70

while time.time() < t_end:
    ret, frame = webcam.read()
    if not ret:
        break

    outputVideoWriter.write(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
outputVideoWriter.release()
cv2.destroyAllWindows()