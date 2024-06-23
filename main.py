from djitellopy import Tello

import cv2
from ultralytics import YOLO
from time import time

import utils

model = YOLO("yolov8s.pt")
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
tracker = utils.create_vit_tracker()


drone = Tello()
drone.connect()
drone.streamon()


def main():
    trackable = False
    detecting = True
    last_seen_x = None
    time_last_seen = 0

    while True:
        frame = drone.get_frame_read().frame
        if frame is None:
            continue

        x, y, w, h = None, None, None, None

        # are we looking for objects?
        if detecting:
            #x, y, w, h = utils.detect_biggest_face(detector, frame)
            x, y, w, h = utils.detect_yolo_object(model, frame, valid_classnames={"sports ball"}, lowest_conf=0.3)
            if x is not None:
                time_last_seen = time()
                tracker.init(frame, (x, y, w, h))
                trackable = True

        # or are we tracking?
        if trackable:
            x, y, w, h = utils.update_tracker(tracker, frame, bounding_box=(x, y, w, h), lowest_allowed_score=0.5,
                                              tracker_is_primary_source=not detecting)
            if x is not None:
                # we still see it
                time_last_seen = time()
            else:
                # if we have not seen our target for >3 seconds, assume it is lost
                if time() > time_last_seen + 3.0:
                    trackable = False  # assume that we lost track

        # 2. display the status info
        status = "DETECTING..." if detecting else "TRACKING..." if trackable else "LOST!"
        cv2.putText(frame, status, (5, 25), cv2.FONT_HERSHEY_PLAIN, 2, utils.GREEN, 2)

        # 3. if already tracking the object, move towards it
        if not detecting and x is not None:
            # if object is detected, set speed towards it
            relative_x, relative_y, relative_width = utils.to_relative_xyw(frame, x, y, w, h)
            utils.follow_object(drone, relative_x, relative_y, relative_width)
            last_seen_x = relative_x
        elif not detecting and last_seen_x is not None:
            # if not detected, but was seen before, slowly turn to where it was last seen
            yaw = +20 if last_seen_x > 0 else -20
            drone.send_rc_control(0, 0, 0, yaw)
        else:
            # if never seen before, hover at the same spot as possible
            drone.send_rc_control(0, 0, 0, 0)

        # show the video frame and listen to what the user is pressing
        cv2.imshow('drone video', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('l'):
            drone.land()  # L = land
        elif key == ord('t'):
            drone.takeoff()  # T = takeoff
        elif key == ord('d'):
            detecting = True  # D = detect
        elif key == ord('f'):
            detecting = False  # F = follow


if __name__ == "__main__":
    main()
