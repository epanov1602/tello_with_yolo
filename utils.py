import cv2
import numpy as np


GREEN = (127, 255, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
PURPLE = (255, 0, 255)


COCO_CLASSNAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def create_vit_tracker():
    """
    Create a tracker that uses VIT
    """
    params = cv2.TrackerVit.Params()
    params.net = "object_tracking_vittrack_2023sep.onnx"
    return cv2.TrackerVit.create(params)


def update_tracker(tracker, frame, bounding_box, lowest_allowed_score, tracker_is_primary_source):
    """
    Update a tracker (with a new videoframe), assuming that the tracker has been initialized to track something
    :param tracker: the tracker to update
    :param frame: new video frame
    :param bounding_box: bounding box of the object we are tracking, in case we approximately know
    :param lowest_allowed_score: if the tracker confidence score is lower than this, do not use its output
    :param tracker_is_primary_source: if tracker_is_primary_source==True, function will return None when unsuccessful
    :return: bounding box in format (x, y, w, h), or if track is lost then (None, None, None, None)
    """
    # will the tracker give us the updated bounding box?
    located, (tx, ty, tw, th) = tracker.update(frame)
    if located and tracker.getTrackingScore() >= lowest_allowed_score:

        # if tracking almost full screen now, let's say tracker diverged
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        diverged = th > 0.75 * frame_height or tw > 0.75 * frame_width

        text = "diverged" if diverged else "trk conf: {:.2}".format(tracker.getTrackingScore())
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), PURPLE, 2)
        cv2.putText(frame, text, (tx + 10, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PURPLE, 2)

        if tracker_is_primary_source and not diverged:
            return tx, ty, tw, th

    # if tracker is not a primary source of bounding box, return the bounding box we had coming in
    if tracker_is_primary_source:
        return None, None, None, None
    else:
        return bounding_box


def detect_biggest_face(face_detector, frame):
    """
    Use HAAR cascade detector to detect faces (picks the widest one, if many found)
    :param frame: a video frame, color or grayscale
    :param face_detector: cascade face detector
    :return: (x, y, w, h) bounding box or (None, None, None, None)
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=9)

    biggest_w = 0  # this will be the width of the biggest face
    for (x, y, w, h) in faces:
        if w > biggest_w:
            biggest_w = w

    for (x, y, w, h) in faces:
        # is this the biggest face seen? green box for it, otherwise red for smaller
        if w == biggest_w:
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, thickness=2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), RED, thickness=2)

    for (x, y, w, h) in faces:
        # is this the biggest face seen? determine the relative X and Y of its center
        if w == biggest_w:
            rel_x, rel_y, rel_w = to_relative_xyw(frame, x, y, w, h)
            text = "rx, ry, rw: {:.2}, {:.2}, {:.3}".format(rel_x, rel_y, rel_w)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, WHITE, 2)
            return x, y, w, h

    # if there were no faces, return (None, None, None, None)
    return None, None, None, None


def detect_yolo_object(yolo_model, frame, valid_classnames=("person", "car"), lowest_conf=0.3):
    """
    Detect an object using a YOLO model (if multiple objects detected, picks the widest)
    :return: either (x, y, w, h) for the bounding box, or (None, None, None, None)
    """
    boxes = []
    biggest_width = 0
    results = yolo_model.predict(frame)

    for result in results:
        for bbox in result.boxes:
            classname = result.names[int(bbox.cls[0])]
            conf = float(bbox.conf)
            x, y, x2, y2 = bbox.xyxy[0]
            if classname in valid_classnames and conf > lowest_conf:
                width, height = int(x2 - x), int(y2 - y)
                boxes.append([int(x), int(y), width, height])
                if width > biggest_width:
                    biggest_width = width
            cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), RED, 2)
            cv2.putText(frame, "{} @ conf={:.2}".format(classname, conf), (int(x) + 10, int(y) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    for (x, y, w, h) in boxes:
        if w == biggest_width:
            cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 4)
            return x, y, w, h

    # otherwise, nothing found
    return None, None, None, None


def to_relative_xyw(frame, x, y, w, h):
    """
    Convert the x, y, width and height of a detected object into [-0.5; +0.5] space,
    so robot can use them in navigation
    :param frame: frame from the camera (colored or grayscale), numpy ndarray
    :param x: X of the upper left corner of the detected object bounding box (in pixels)
    :param y: Y of the upper left corner of the detected object bounding box (in pixels)
    :param w: width of the detected object box (in pixels)
    :param h: height of the detected object box (in pixels)
    :return: X, Y, W remapped to [-0.5; +0.5] space -- where X, Y are center of the object
    """
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    relative_x = (x + w // 2) / frame_width - 0.5  # can be between -0.5 and +0.5
    relative_y = (y + h // 2) / frame_height - 0.5  # can be between -0.5 and +0.5
    relative_y = -relative_y  # flip the sign, so that when object is above Y is positive
    relative_width = w / frame_width
    return relative_x, relative_y, relative_width


def follow_object(drone, relative_x, relative_y, relative_width, target_width=0.1667, target_y=0.0):
    """
    Sets Tello speed to follow the detected object, using a very primitive PID logic
    :param drone: a djitellopy.Tello
    :param relative_x: output of to_relative_xyw(...)
    :param relative_y: output of to_relative_xyw(...)
    :param relative_width: output of to_relative_xyw(...)
    :param target_width: how wide should the object be for the drone to stop approaching it
    :param target_y: set to -0.1 if you want the drone to stay slightly above the target, +0.1 for slightly below
    :return: None
    """
    # turn right if X>0, turn left if X<0
    yaw = relative_x * 200
    # ^^ this way if object is at -0.3, we turn with speed -0.60; but if object is too close, we turn slowly

    # go up if Y>0, go down if Y<0
    up = (relative_y - target_y) * 100  # this way if object is at -0.1, we move at speed -10

    forward, left = 0, 0
    if relative_width < 0.01:
        forward = 100  # full forward, we are very far from our object
    else:
        # if relative_width > target, we are too close => move back (for example, speed=-35 if relative_width=100%)
        # if relative_width < target, we are too far => move forward (for example, speed=96 if relative_width=5%)
        forward = 7.0 * (1 / relative_width - 1 / target_width)

    drone.send_rc_control(
        int(left),
        int(np.clip(forward, -100, +100)),
        int(np.clip(up, -100, +100)),
        int(np.clip(yaw, -100, +100))
    )

