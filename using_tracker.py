import cv2
from time import time
from ultralytics import YOLO
import utils

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
tracker = utils.create_vit_tracker()
model = YOLO("yolov8s.pt")


camera = cv2.VideoCapture(0)
success, frame = camera.read()


trackable = False
detecting = True
time_last_seen = 0


while True:
    success, frame = camera.read()
    if not success:
        continue

    x, y, w, h = None, None, None, None

    if detecting:
        x, y, w, h = utils.detect_biggest_face(detector, frame)
        # x, y, w, h = utils.detect_yolo_object(model, frame, valid_classnames={"sports ball"}, lowest_conf=0.3)
        if x is not None:
            time_last_seen = time()
            tracker.init(frame, (x, y, w, h))
            trackable = True

    if trackable:
        x, y, w, h = utils.update_tracker(tracker, frame, bounding_box=(x, y, w, h), lowest_allowed_score=0.5, tracker_is_primary_source=not detecting)
        if x is not None:
            # we still see it
            time_last_seen = time()
        else:
            # if we have not seen our target for >3 seconds, assume it is lost
            if time() > time_last_seen + 3.0:
                trackable = False  # assume that we lost track

    status = "DETECTING..." if detecting else "TRACKING..."
    cv2.putText(frame, status, (5, 25), cv2.FONT_HERSHEY_PLAIN, 2, utils.GREEN, 2)
    cv2.imshow("camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        detecting = True
    elif key == ord('t'):
        detecting = False

