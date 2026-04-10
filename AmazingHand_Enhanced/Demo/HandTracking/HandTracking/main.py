import os
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
import pyarrow as pa
from dora import Node

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

# Number of frames to collect during calibration
CALIB_FRAMES = 30


@dataclass
class CalibResult:
    # X offset for tip1, tip2, tip3 (index [0] of each tip in the hand frame).
    # Measured during open-hand calibration and subtracted during tracking,
    # so that the three finger tips read X=0 when the hand is fully open and flat.
    x_offsets: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


def _draw_text_bg(
    img, text, pos, font_scale=0.7, color=(255, 255, 255), bg=(0, 0, 0), thickness=2
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def process_img(hand_proc, image, calib=None):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hand_proc.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    r_res = None
    l_res = None
    if results.multi_hand_landmarks:

        for index, handedness_classif in enumerate(results.multi_handedness):
            if (
                handedness_classif.classification[0].score > 0.8
            ):  # let's considere only one right hand

                hand_landmarks = results.multi_hand_world_landmarks[index]  # metric
                hand_landmarks_norm = results.multi_hand_landmarks[index]  # normalized

                # Wrist as origin
                origin = np.array(
                    [
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z,
                    ]
                )

                # Creating coordinate system
                mid_mcp = np.array(
                    [
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].x,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].y,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                        ].z,
                    ]
                )  # base of the middle finger
                index_mcp = np.array(
                    [
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].x,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].y,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_MCP
                        ].z,
                    ]
                )  # base of the index finger
                ring_mcp = np.array(
                    [
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.RING_FINGER_MCP
                        ].x,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.RING_FINGER_MCP
                        ].y,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.RING_FINGER_MCP
                        ].z,
                    ]
                )  # base of the ring finger

                unit_z = (
                    mid_mcp - origin
                )  # z is unit vector from base of wrist toward base of middle finger
                unit_z = unit_z / np.linalg.norm(unit_z)

                if handedness_classif.classification[0].label == "Right":
                    vec_towards_y = (
                        origin - index_mcp
                    )  # vector from wrist base towards index base
                if handedness_classif.classification[0].label == "Left":
                    vec_towards_y = (
                        index_mcp - origin
                    )  # vector from wrist base towards index base

                unit_x = np.cross(
                    vec_towards_y, unit_z
                )  # we say unit x is the cross product of z and the vector towards pinky

                unit_x = unit_x / np.linalg.norm(unit_x)

                unit_y = np.cross(unit_z, unit_x)

                R = np.array([unit_x, -unit_y, unit_z]).reshape(
                    (3, 3)
                )  # -y because of mirror?

                # Get tip position in world coordinates

                tip1_world = np.array(
                    [
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ].x,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ].y,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.INDEX_FINGER_TIP
                        ].z,
                    ]
                )

                tip2_world = np.array(
                    [
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                        ].x,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                        ].y,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                        ].z
                        + 0.04,
                    ]
                )

                tip3_world = np.array(
                    [
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.RING_FINGER_TIP
                        ].x,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.RING_FINGER_TIP
                        ].y,
                        hand_landmarks.landmark[
                            mp_hands.HandLandmark.RING_FINGER_TIP
                        ].z,
                    ]
                )

                tip4_world = np.array(
                    [
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z,
                    ]
                )

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks_norm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                tip1 = R @ (tip1_world - origin)
                tip2 = R @ (tip2_world - origin)
                tip3 = R @ (tip3_world - origin)
                tip4 = R @ (tip4_world - origin)

                middle_finger_mcp = R @ (mid_mcp - origin)
                index_finger_mcp = R @ (index_mcp - origin)
                ring_finger_mcp = R @ (ring_mcp - origin)

                if handedness_classif.classification[0].label == "Right":
                    if calib:
                        # Tip can not be behind MCP if the finger is totally bent
                        if tip1[2] - index_finger_mcp[2] < 0:
                            tip1[0] = max(tip1[0], calib["r"].x_offsets[0])
                        if tip2[2] - middle_finger_mcp[2] < 0:
                            tip2[0] = max(tip2[0], calib["r"].x_offsets[1])
                        if tip3[2] - ring_finger_mcp[2] < 0:
                            tip3[0] = max(tip3[0], calib["r"].x_offsets[2])
                    r_res = [
                        {"r_tip1": tip1, "r_tip2": tip2, "r_tip3": tip3, "r_tip4": tip4}
                    ]
                elif handedness_classif.classification[0].label == "Left":
                    if calib:
                        # Tip can not be behind MCP if the finger is totally bent
                        if tip1[2] - index_finger_mcp[2] < 0:
                            tip1[0] = max(tip1[0], calib["l"].x_offsets[0])
                        if tip2[2] - middle_finger_mcp[2] < 0:
                            tip2[0] = max(tip2[0], calib["l"].x_offsets[1])
                        if tip3[2] - ring_finger_mcp[2] < 0:
                            tip3[0] = max(tip3[0], calib["l"].x_offsets[2])
                    l_res = [
                        {"l_tip1": tip1, "l_tip2": tip2, "l_tip3": tip3, "l_tip4": tip4}
                    ]
    # Flip the image horizontally for a selfie-view display.
    return image, r_res, l_res


def run_calibration(cap, hands):
    """Calibrate both hands simultaneously.

    The user opens both hands fully and presses SPACE.  Frames are collected
    for each hand independently; a hand whose history resets (lost detection)
    doesn't affect the other.  The loop ends when both hands have reached
    CALIB_FRAMES valid frames.

    Returns a dict {"r": CalibResult, "l": CalibResult}.
    Press Q to skip entirely (identity offsets for both hands).
    """
    print("=== CALIBRATION ===")
    print("Open both hands fully and press SPACE to start.")
    print("Press Q to skip.")

    r_history = []  # list of (4, 3) arrays for right hand
    l_history = []  # list of (4, 3) arrays for left hand
    collecting = False

    r_tip_keys = ["r_tip1", "r_tip2", "r_tip3", "r_tip4"]
    l_tip_keys = ["l_tip1", "l_tip2", "l_tip3", "l_tip4"]

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame, r_res, l_res = process_img(hands, frame)
        h, w = frame.shape[:2]

        if not collecting:
            _draw_text_bg(
                frame, "=== HAND CALIBRATION ===", (20, 45), 0.9, (0, 255, 255)
            )
            _draw_text_bg(
                frame, "Open both hands fully (palms facing camera)", (20, 90)
            )
            _draw_text_bg(frame, "Press SPACE to start", (20, 130))
            _draw_text_bg(frame, "Press Q to skip", (20, 170), color=(160, 160, 160))
        else:
            bar_w = int(w * 0.4)
            gap = int(w * 0.1)
            bar_y = h // 2 - 15

            for label, history, res, tip_keys, bar_x in (
                ("LEFT", l_history, l_res, l_tip_keys, gap),
                ("RIGHT", r_history, r_res, r_tip_keys, gap + bar_w + gap),
            ):
                hand_complete = res is not None and all(k in res[0] for k in tip_keys)
                if hand_complete:
                    tips = np.array([res[0][k] for k in tip_keys])
                    history.append(tips)

                progress = min(len(history) / CALIB_FRAMES, 1.0)
                done = len(history) >= CALIB_FRAMES

                bar_color = (0, 160, 255) if not done else (0, 200, 0)
                cv2.rectangle(
                    frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 30), (40, 40, 40), -1
                )
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + int(bar_w * progress), bar_y + 30),
                    bar_color,
                    -1,
                )
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + bar_w, bar_y + 30),
                    (255, 255, 255),
                    2,
                )
                status = "OK" if done else f"{int(progress * 100)}%"
                _draw_text_bg(
                    frame,
                    f"{label}: {status}",
                    (bar_x, bar_y - 10),
                    0.7,
                    (0, 160, 255) if not done else (0, 255, 0),
                )

                if not hand_complete and not done:
                    history.clear()  # reset only this hand on loss

            if len(r_history) >= CALIB_FRAMES and len(l_history) >= CALIB_FRAMES:
                break

        cv2.imshow("MediaPipe Hands", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Calibration skipped - using identity corrections.")
            return {"r": CalibResult(), "l": CalibResult()}
        if key == ord(" ") and not collecting:
            collecting = True
            r_history.clear()
            l_history.clear()

    def _compute(history):
        mean_tips = np.mean(np.array(history), axis=0)  # (4, 3)
        return CalibResult(x_offsets=mean_tips[:3, 0].tolist())

    r_calib = _compute(r_history)
    l_calib = _compute(l_history)

    print(
        f"Calibration complete - R offsets: {[f'{v:.4f}' for v in r_calib.x_offsets]}"
    )
    print(
        f"Calibration complete - L offsets: {[f'{v:.4f}' for v in l_calib.x_offsets]}"
    )

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        _draw_text_bg(frame, "Calibration complete!", (20, 45), 0.9, (0, 255, 0))
        _draw_text_bg(
            frame, f"R offsets: {[f'{v:.3f}' for v in r_calib.x_offsets]}", (20, 90)
        )
        _draw_text_bg(
            frame, f"L offsets: {[f'{v:.3f}' for v in l_calib.x_offsets]}", (20, 130)
        )
        _draw_text_bg(frame, "Starting tracking...", (20, 170), color=(200, 200, 200))
        cv2.imshow("MediaPipe Hands", frame)
        cv2.waitKey(2000)

    return {"r": r_calib, "l": l_calib}


def apply_calibration(res, calib, prefix):
    """Subtract per-finger X offsets from tip1, tip2, tip3.

    During calibration the hand is fully open and we record the X coordinate
    of each of the three fingertips.  Subtracting those values during tracking
    centres X=0 at the open-hand reference position for each finger.
    tip4 (thumb) is left unchanged.

    Args:
        res:    Raw output of process_img for one hand (list of one dict, or None).
        calib:  CalibResult computed during the calibration phase.
        prefix: 'r' for right hand, 'l' for left hand.

    Returns:
        Corrected result in the same format, or None.
    """
    if res is None:
        return None
    corrected = dict(res[0])
    for i, key in enumerate(f"{prefix}_tip{n}" for n in (1, 2, 3)):
        if key in corrected:
            tip = corrected[key].copy()
            tip[0] -= calib.x_offsets[i]
            corrected[key] = tip
    return [corrected]


def main():

    node = Node()

    pa.array([])  # initialize pyarrow array
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:

        # --- Calibration phase (runs before the dora event loop) ---
        calib = run_calibration(cap, hands)

        for event in node:

            event_type = event["type"]

            if event_type == "INPUT":
                event_id = event["id"]

                if event_id == "tick":
                    ret, frame = cap.read()

                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    # process
                    frame, r_res, l_res = process_img(hands, frame, calib)

                    # Apply calibration corrections before sending
                    r_res = apply_calibration(r_res, calib["r"], "r")
                    l_res = apply_calibration(l_res, calib["l"], "l")

                    if r_res is not None:
                        node.send_output("r_hand_pos", pa.array(r_res))
                    if l_res is not None:
                        node.send_output("l_hand_pos", pa.array(l_res))

                    cv2.imshow("MediaPipe Hands", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            elif event_type == "ERROR":
                raise RuntimeError(event["error"])


if __name__ == "__main__":
    main()
