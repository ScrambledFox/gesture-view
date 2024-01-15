#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from utils import draw_landmarks
from model import KeyPointClassifier
from model import PointHistoryClassifier

EVENT_TIMER_MAX = 1000

# Enum for interaction zones
INTERACTION_ZONE = {
    "BOTTOM": 0,
    "CENTER": 1,
    "TOP": 2,
}

COLOUR_SELECTION_STRATEGY = {
    "SWIPE": 0,
    "POINT": 1,
    "THUMBS_DOWN": 2,
}

# offsetLimits = [30, 400]
colorRadius = 20 * 2
colorSpacing = 15
yAlign = 50


# Shopping chart state
shoppingChartCount = 0

COLOURS = [
    (0, 0, 255),  # red 0
    (0, 255, 255),  # yellow 1
    (0, 255, 0),  # green 2
    (255, 255, 0),  # cyan 3
    (255, 0, 0),  # blue 4
    (255, 0, 255),  # magenta 5
]

DRAW_DEBUG_UI = True


# POINT AND SWIPE
# POINT
# THUMBS DOWN TO CYCLE

# OK TO ADD TO CART
# POINT TO CART TO ADD TO CART
# THUMBS UP TO ADD TO CART


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=int,
        default=0.5,
    )

    args = parser.parse_args()

    return args


def main():
    # Globals fuck python
    global shoppingChartCount

    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Load images ################
    shoppingChart = cv.imread("chart.png")

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    with open(
        "model/point_history_classifier/point_history_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #####################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)
    lastFingerPos = [0, 0]

    #  ########################################################################
    mode = 0

    # Timed gesture event ###################################################
    currentGestureId = -1
    currentHistoryGestureId = -1
    eventTimer = 0

    # Motion gestures
    interaction_start_x = 0
    leftOffsetSinceInteractionStart = 30

    # Colour selection
    selectedColour = 0
    colourSelectionMode = COLOUR_SELECTION_STRATEGY["SWIPE"]
    leftOffset = 30

    # Callibration
    uiPoints = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    # Event queue
    eventQueue = deque(maxlen=10)

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        number, mode = select_mode(key, mode)
        colourSelectionMode = select_colour_selection_mode(key, colourSelectionMode)
        uiPoints = callibrate_ui_points(key, uiPoints, lastFingerPos)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        debug_image, selectedColour = draw_ui(
            debug_image, leftOffset, selectedColour, colourSelectionMode
        )

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history
                )
                # Write to the dataset file
                logging_csv(
                    number,
                    mode,
                    pre_processed_landmark_list,
                    pre_processed_point_history_list,
                )

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list
                    )

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # Process gestures
                (
                    eventQueue,
                    eventTimer,
                    currentGestureId,
                    currentHistoryGestureId,
                ) = process_gesture(
                    eventQueue,
                    currentGestureId,
                    currentHistoryGestureId,
                    hand_sign_id,
                    finger_gesture_id,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[finger_gesture_id],
                    selectedColour,
                    colourSelectionMode,
                    brect,
                    eventTimer,
                    fps,
                )

                # Motion gestures
                # Point gesture
                if hand_sign_id == 2:
                    lastFingerPos = [brect[1], brect[0]]

                if (
                    hand_sign_id == 2
                    and colourSelectionMode == COLOUR_SELECTION_STRATEGY["SWIPE"]
                ):
                    # only assign the first time
                    if interaction_start_x == -1:
                        interaction_start_x = brect[1]
                        leftOffsetSinceInteractionStart = leftOffset

                    leftOffset, lastFingerPos = handle_point_gesture_event(
                        interaction_start_x,
                        leftOffsetSinceInteractionStart,
                        brect,
                        lastFingerPos,
                    )
                else:
                    interaction_start_x = -1

                # Handle all Events in queue
                while len(eventQueue) > 0:
                    event = eventQueue.popleft()
                    if event[0] == "ShiftColour":
                        print("shift colour event")
                        selectedColour = (selectedColour + 1) % 6
                        print(selectedColour)

                    elif event[0] == "PointStopEvent":
                        print("point stop event")
                        pointedUi = get_pointed_ui(
                            event[1], event[2], uiPoints
                        )

                        # Color selection
                        if pointedUi > 0 and pointedUi < 6:
                            selectedColour = pointedUi
                            print(selectedColour)
                        # Shopping chart selection
                        elif pointedUi == 6:
                            shoppingChartCount = shoppingChartCount + 1


                # Drawing part
                if DRAW_DEBUG_UI:
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        point_history_classifier_labels[most_common_fg_id[0][0]],
                    )
        else:
            point_history.append([0, 0])

        if DRAW_DEBUG_UI:
            debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_info(debug_image, fps, mode, colourSelectionMode, number)

        cv.namedWindow("Smart mirror", cv.WINDOW_NORMAL)
        cv.setWindowProperty(
            "Smart mirror", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN
        )

        # Add shopping chart
        shoppingChartWithCount = shoppingChart.copy()
        cv.putText(
        shoppingChartWithCount,
        str(shoppingChartCount),
        (20, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv.LINE_AA,
        )
        shoppingChartRotated = cv.rotate(shoppingChartWithCount, cv.ROTATE_90_CLOCKWISE)
        debug_image = merge_image(debug_image, shoppingChartRotated, 960-80, 540-50)

        # Screen reflection #############################################################
        cv.imshow("Smart mirror", debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def select_colour_selection_mode(key, colourSelectionMode):
    if key == 115:  # s
        colourSelectionMode = COLOUR_SELECTION_STRATEGY["SWIPE"]
    if key == 112:  # p
        colourSelectionMode = COLOUR_SELECTION_STRATEGY["POINT"]
    if key == 116:  # t
        colourSelectionMode = COLOUR_SELECTION_STRATEGY["THUMBS_DOWN"]
    return colourSelectionMode


def callibrate_ui_points(key, uiPoints, lastFingerPos):
    if key == 49:
        uiPoints[0] = lastFingerPos
        print("Callibrated ui point 0: " + str(uiPoints[0]))
    if key == 50:
        uiPoints[1] = lastFingerPos
        print("Callibrated ui point 1: " + str(uiPoints[1]))
    if key == 51:
        uiPoints[2] = lastFingerPos
        print("Callibrated ui point 2: " + str(uiPoints[2]))
    if key == 52:
        uiPoints[3] = lastFingerPos
        print("Callibrated ui point 3: " + str(uiPoints[3]))
    if key == 53:
        uiPoints[4] = lastFingerPos
        print("Callibrated ui point 4: " + str(uiPoints[4]))
    if key == 54:
        uiPoints[5] = lastFingerPos
        print("Callibrated ui point 5: " + str(uiPoints[5]))
    if key == 55:
        uiPoints[6] = lastFingerPos
        print("Callibrated ui point 6: " + str(uiPoints[6]))

    return uiPoints

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def get_interaction_zone(brect):
    interactionZone = INTERACTION_ZONE["CENTER"]
    if brect[0] > 600:
        interactionZone = INTERACTION_ZONE["TOP"]
    elif brect[0] < 300:
        interactionZone = INTERACTION_ZONE["BOTTOM"]

    return interactionZone


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (
            temp_point_history[index][0] - base_x
        ) / image_width
        temp_point_history[index][1] = (
            temp_point_history[index][1] - base_y
        ) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


# If the current gesture is held for a certain period of time,
# it is recognized as a gesture event
def process_gesture(
    eventQueue,
    currentGestureId,
    currentFingerGestureId,
    detectedGestureId,
    detectedFingerGestureId,
    detectedGestureLabel,
    detectedFingerGestureLabel,
    selectedColour,
    currentInteractionStrategy,
    brect,
    eventTimer,
    fps,
):
    # Get interaction zone from bounding rect
    interactionZone = get_interaction_zone(brect)

    if detectedGestureId == 2:  # Point gesture
        if currentFingerGestureId == detectedFingerGestureId:
            eventTimer += 1000 / fps
            if eventTimer > EVENT_TIMER_MAX:
                eventTimer = 0
                eventQueue = handle_motion_gesture_event(
                    eventQueue,
                    detectedFingerGestureId,
                    detectedFingerGestureLabel,
                    currentInteractionStrategy,
                    interactionZone,
                    brect,
                )
        else:
            currentFingerGestureId = detectedFingerGestureId
            eventTimer = 0

        return (
            eventQueue,
            eventTimer,
            currentGestureId,
            currentFingerGestureId,
        )

    else:
        if currentGestureId == detectedGestureId:
            eventTimer += 1000 / fps
            if eventTimer > EVENT_TIMER_MAX:
                eventTimer = 0
                eventQueue = handle_gesture_event(
                    eventQueue,
                    detectedGestureId,
                    detectedGestureLabel,
                    currentInteractionStrategy,
                    interactionZone,
                )
        else:
            currentGestureId = detectedGestureId
            eventTimer = 0

        return (
            eventQueue,
            eventTimer,
            currentGestureId,
            currentFingerGestureId,
        )


def handle_gesture_event(eventQueue, id, label, interactionStrategy, interactionZone):
    print("Gesture Event Detected:" + str(label) + ":" + str(interactionZone))

    if interactionStrategy == COLOUR_SELECTION_STRATEGY["THUMBS_DOWN"]:
        if label == "ThumbsDown":
            print("shift colour")
            eventQueue.append(["ShiftColour"])
        if label == "ThumbsUp":
            print("add to chart")
            global shoppingChartCount
            shoppingChartCount = shoppingChartCount+1

    return eventQueue


def handle_motion_gesture_event(
    eventQueue, id, label, interactionStrategy, interactionZone, coords
):
    print("Motion Gesture Event Detected:" + str(label) + ":" + str(interactionZone))

    if interactionStrategy == COLOUR_SELECTION_STRATEGY["POINT"]:
        if label == "Stop":
            print("Stop")
            eventQueue.append(["PointStopEvent", interactionZone, coords])

    return eventQueue


def handle_point_gesture_event(
    interaction_start_x, leftOffsetSinceInteractionStart, brect, lastFingerPos
):
    deltaX = interaction_start_x - brect[1]
    print("Point Gesture Event Detected:" + str(deltaX))
    return leftOffsetSinceInteractionStart - deltaX, lastFingerPos


# Go through the ui points and find the closest one and return the index,
# if none are close enough return -1
def get_pointed_ui(interactionZone, brect, uiPoints):
    # if interactionZone != INTERACTION_ZONE["BOTTOM"]:
    #     return -1

    print(brect)
    print(interactionZone)

    coords = [brect[1], brect[0]]

    closestUi = -1
    closest = 100
    for i in range(7):
        dist = np.linalg.norm(np.array(coords) - np.array(uiPoints[i]))
        print(
            "Distance between "
            + str(coords)
            + " and "
            + str(uiPoints[i])
            + " is "
            + str(dist)
        )
        if dist < closest:
            closest = dist
            closestUi = i

    return closestUi


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = "model/point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    # Display coords of brect
    cv.putText(
        image,
        str(brect[0]) + "," + str(brect[1]),
        (brect[0] + 5, brect[3] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    if finger_gesture_text != "":
        cv.putText(
            image,
            "Finger Gesture:" + finger_gesture_text,
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            "Finger Gesture:" + finger_gesture_text,
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(
                image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2
            )

    return image


def draw_ui(image, leftOffset, selectedColour, colourSelectionMode):
    if colourSelectionMode != COLOUR_SELECTION_STRATEGY["SWIPE"]:
        leftOffset = 120

    rX = colorRadius + leftOffset
    mX = rX + colorRadius + colorSpacing
    bX = mX + colorRadius + colorSpacing
    cX = bX + colorRadius + colorSpacing
    gX = cX + colorRadius + colorSpacing
    yX = gX + colorRadius + colorSpacing

    positions = [rX, mX, bX, cX, gX, yX]

    if colourSelectionMode == COLOUR_SELECTION_STRATEGY["SWIPE"]:
        selectedColour = positions.index(min(positions, key=lambda x: abs(x - 270)))
    elif colourSelectionMode == COLOUR_SELECTION_STRATEGY["POINT"]:
        pass
    elif colourSelectionMode == COLOUR_SELECTION_STRATEGY["THUMBS_DOWN"]:
        pass

    upperLimit = np.array([255, 255, 255])
    lowerLimit = np.array([200, 200, 200])
    mask = cv.inRange(image, lowerLimit, upperLimit)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largestContour = max(contours, key=cv.contourArea)
        imageWithContours = copy.deepcopy(image)
        imageWithContours = cv.drawContours(
            imageWithContours,
            [largestContour],
            0,
            COLOURS[selectedColour],
            thickness=cv.FILLED,
        )
        alpha = 0.5
        image = cv.addWeighted(image, 1 - alpha, imageWithContours, alpha, 0)

    for i in range(6):
        size = 20
        borderColor = (0, 0, 0)
        if selectedColour == i:
            size = size + 10
            borderColor = (255, 255, 255)
        image = cv.circle(image, (yAlign, positions[i]), size + 2, borderColor, -1)
        image = cv.circle(image, (yAlign, positions[i]), size, COLOURS[i], -1)

    return image, selectedColour

def merge_image(back, front, x,y):
    # convert to rgba
    if back.shape[2] == 3:
        back = cv.cvtColor(back, cv.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv.cvtColor(front, cv.COLOR_BGR2BGRA)

    # crop the overlay from both images
    bh,bw = back.shape[:2]
    fh,fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x+fw, bw)
    y1, y2 = max(y, 0), min(y+fh, bh)
    front_cropped = front[y1-y:y2-y, x1-x:x2-x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:,:,3:4] / 255
    alpha_back = back_cropped[:,:,3:4] / 255
    
    # replace an area in result with overlay
    result = back.copy()
    print(f'af: {alpha_front.shape}\nab: {alpha_back.shape}\nfront_cropped: {front_cropped.shape}\nback_cropped: {back_cropped.shape}')
    result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:,:,:3] + (1-alpha_front) * back_cropped[:,:,:3]
    result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front*alpha_back) * 255

    return result

def draw_info(image, fps, mode, interactionMode, number):
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        "FPS:" + str(fps),
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    cv.putText(
        image,
        "Interaction Mode: " + str(interactionMode),
        (10, 140),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    mode_string = ["Logging Key Point", "Logging Point History"]
    if 1 <= mode <= 2:
        cv.putText(
            image,
            "MODE:" + mode_string[mode - 1],
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= number <= 9:
            cv.putText(
                image,
                "NUM:" + str(number),
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    return image


if __name__ == "__main__":
    main()
