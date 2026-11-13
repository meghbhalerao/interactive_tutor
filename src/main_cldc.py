import argparse
import hydra
import cv2
import pickle
import numpy as np
from omegaconf import DictConfig
import os
import sys
import json
import time
import src.domains.cldc.geometry as gm
from src.domains.cldc.sterile_field import SterileField
from src.reader import Reader
from src.utils.mediapipe_utils import MediaPipe
from domains.cldc.error_categorization import error_category

# below are some config parameters
here = os.path.dirname(os.path.abspath(__file__))
DEPTH_VIS_MAX = 4.0  # m; the realtime depth video maps 0..DEPTH_VIS_MAX linearly onto the colormap, farther is clipped
model_path = "/home/mb230/projects/adaptive_skill/assets/mediapipe_models/pose_landmarker.task"
hand_model_path = "/home/mb230/projects/adaptive_skill/assets/mediapipe_models/hand_landmarker.task"
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# phase is one of: 'realtime' (live camera) or 'offline_mp4' (recorded mp4 videos); both run pose estimation + rule scoring and save an overlay video
phase = 'realtime'
save_pose_dump = False  # also save per-frame landmarks to pose.npz

# 'verbose' draws the skeletal tracks and rule violations on the display; 'silent' shows the live feed without them
parser = argparse.ArgumentParser()
parser.add_argument("--notification", choices=["verbose", "silent"], default="verbose")
parser.add_argument("--phase", choices=["realtime", "offline_mp4"], default="realtime")
parser.add_argument("--save-dir", type=str, required=True)
parser.add_argument("--error-cat-label-mode", type=str, required=True)
args = parser.parse_args()

notification = args.notification
phase = args.phase 
# path to the directory of the recording
save_dir = args.save_dir
error_cat_label_mode = args.error_cat_label_mode

# (index into sterile_field.is_contaminated output, on-screen label)
RULES = [(1, "Hands below waistline"), (2, "Reaching over sterile field"), (3, "Back turned on sterile field"), (4, "Touching one inch border")]
rule_to_idx = {"hbt": 1, "ra": 2, "tb": 3, "oib": 4}
idx_to_rule = {1: "hbt", 2: "ra", 3: "tb", 4: "oib"}

os.makedirs(save_dir, exist_ok=True)

error_dict = {}

def draw_rules(frame_bgr, sterile_field, human, notification):
    """Overlay 'human not found', or one line per rule (red = violated, green = ok) with the number it thresholds next to it."""
    if notification == 'verbose':
        if not human.human_found:
            cv2.putText(frame_bgr, "human not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            return
        for row, (idx, label) in enumerate(RULES):
            violated = sterile_field.contaminations[idx]
            text = f"{'VIOLATED' if violated else 'ok'}: {label} = {sterile_field.rule_values[idx]:+.3f}"
            cv2.putText(frame_bgr, text, (10, 30 * (row + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if violated else (0, 160, 0), 2)
    elif notification == 'silent':
        pass
    else:
        raise ValueError(f"Notification type {notification} not defined!")
    return sterile_field.contaminations


def colorize_depth(depth_m):
    """(H, W) depth in m -> (H, W, 3) BGR image, JET over 0..DEPTH_VIS_MAX; pixels with no measurement are black."""
    gray = np.clip(depth_m / DEPTH_VIS_MAX * 255, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    out[depth_m <= 0] = 0
    return out


# the two phases differ only in where frames come from (and whether a window is shown)
if phase == 'realtime':
    reader = Reader('realtime')
    sterile_field = SterileField()
    sterile_field.detect_manual('realtime', 'rgb', reader=reader)
    # save sterile field
    sterile_field_path = os.path.join(save_dir, "sterile_field.pkl")
    with open(sterile_field_path, "wb") as file:
        pickle.dump(sterile_field, file)
    print("sterile field created successfully!" + (" Press q in the video window to quit." if phase == 'realtime' else ""))
    print(f"saved sterile field at {sterile_field_path}")

elif phase == 'offline_mp4':
    reader = Reader(phase, path=save_dir)
    sterile_field_load_path = os.path.join(save_dir, "sterile_field.pkl")
    sterile_field = pickle.load(open(sterile_field_load_path, 'rb'))
    print(f"loaded sterile field from {sterile_field_load_path}")
    print(f"Total number of frames in the reader is {reader.num_frames()}")

else:
    raise ValueError(f"Phase {phase} not implemented/defined yet!")


frames = reader.frames()

# click the 4 sterile-field corners on the first frame
human = MediaPipe(model_path, hand_model_path, min_detection_confidence, min_tracking_confidence)

# one folder per rule; violating frames are saved as <rule>/<frame_idx>.png
out_path = os.path.join(save_dir, "pose.npz")

for rule in rule_to_idx:
    os.makedirs(os.path.join(save_dir, rule), exist_ok=True)

video_path = os.path.join(save_dir, f"{phase}.mp4")

fps = 30 if phase == 'realtime' else 15  
viol_frames = {rule: 0 for rule in rule_to_idx}
depth_frames = []
num_detected = 0
writer = None
raw_writer = None
depth_writer = None
ts, lms_cam = [], []  # only filled if save_pose_dump

print("Total number of frames is {}")


if phase == 'realtime':
    cv2.namedWindow("realtime", cv2.WINDOW_NORMAL)  # WINDOW_NORMAL lets you drag the window corners to resize
    # saved once, before the loop; Reader's height_color/width_color are swapped, so the frame is (width=height_color, height=width_color)
    with open(os.path.join(save_dir, "realtime_depth_info.json"), "w") as f:
        json.dump(dict(fx=reader.new_camera_matrix[0, 0], fy=reader.new_camera_matrix[1, 1], ppx=reader.new_camera_matrix[0, 2], ppy=reader.new_camera_matrix[1, 2], width=reader.height_color, height=reader.width_color, distortion="none (color and depth both undistorted with new_camera_matrix)", depth_scale_m_per_count=reader.depth_scale, colormap="JET", colormap_range_m=[0.0, DEPTH_VIS_MAX], no_measurement="black (0,0,0)"), f, indent=2)
    # frame size from the reader (swapped, see above); the raw and depth videos share size and fps, so frame i of both is the same instant and pixel
    w, h = reader.height_color, reader.width_color
    raw_writer = cv2.VideoWriter(os.path.join(save_dir, "realtime_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    depth_writer = cv2.VideoWriter(os.path.join(save_dir, "realtime_depth.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
try:
    for i, (_, frame_bgr, depth_raw) in enumerate(frames):
        if phase != 'realtime':  # realtime already got w, h from the reader above
            h, w = frame_bgr.shape[:2]
        t_ms = int(i * 1000 / fps)  # elapsed time from frame index; strictly increasing for MediaPipe as long as fps <= 1000
        human.estimate_pose(frame_bgr, depth_raw, reader, t_ms)
        human.estimate_hands(frame_bgr, depth_raw, reader, t_ms)
        if save_pose_dump:
            ts.append(t_ms)
            lms_cam.append(np.array([np.full(3, np.nan) if p is None else p.unpack() for p in human.pose_keypoints_camera_coordinates]))

        if phase == 'realtime':  # save the untouched frame before any overlay is drawn
            raw_writer.write(frame_bgr)
            depth_writer.write(colorize_depth(depth_raw * reader.depth_scale))
            depth_frames.append(depth_raw)

        if notification == 'verbose':
            human.draw(frame_bgr)

        violated = []
        if human.human_found:
            contaminations = sterile_field.is_contaminated(human)
            num_detected += 1
            for rule, idx in rule_to_idx.items():
                if contaminations[idx]:
                    viol_frames[rule] += 1
                    violated.append(rule)

            # this only records conditions where the human and hands are found
            # (contaminations are np.bool_, so use truthiness, not `is True/False`)
            if not contaminations[0]:
                error_dict[i] = {}
                for rule, idx in rule_to_idx.items():
                    if contaminations[idx]:
                        error_dict[i][rule] = error_category(human, rule, error_cat_label_mode = error_cat_label_mode)

        draw_rules(frame_bgr, sterile_field, human, notification)

        if phase == 'realtime':  # outer corners you clicked (red), computed one-inch inner border (black)
            for corners, color in ((sterile_field.out_corners_camera_coordinates, (0, 0, 255)), (sterile_field.in_corners_camera_coordinates, (0, 0, 0))):
                px = [gm.projectPoint(p, np.eye(3), np.zeros(3), reader.new_camera_matrix).unpack() for p in corners]
                for k in range(4):
                    cv2.line(frame_bgr, px[k], px[(k + 1) % 4], color, 1)

        for rule in violated:
            cv2.imwrite(os.path.join(save_dir, rule, f"{i}.png"), frame_bgr)
        if writer is None:
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame_bgr)

        if phase == 'realtime':
            cv2.imshow("realtime", frame_bgr)
            if cv2.waitKey(1) == ord('q'):
                break
finally:
    if num_detected:
        scores = {k: 1 - v / num_detected for k, v in viol_frames.items()}
        print(f"Frames with human detected: {num_detected}")
        print(f"Number of frames violated for each rule is {viol_frames}")
        print(f"Scores are {scores}")
    human.close()
    if writer is not None:
        writer.release()
        print(f"Saved video to {video_path}")
    if raw_writer is not None:
        raw_writer.release()
    if depth_writer is not None:
        depth_writer.release()
    if save_pose_dump and ts:
        np.savez(
            out_path,
            timestamps_ms=np.array(ts),
            landmarks_cam_m=np.stack(lms_cam))     # (T, 33, 3): X, Y, Z in meters
        print(f"Saved pose dump to {out_path}")

    error_dict_path = os.path.join(save_dir, f'error_dict_{error_cat_label_mode}.pkl')
    with open(error_dict_path, 'wb') as f:
        pickle.dump(error_dict, f)
    print(f"Saved error dict to {error_dict_path}")

    if phase == 'realtime':
        depth_frames = np.array(depth_frames)
        depth_path = os.path.join(save_dir, 'depth_vals.npy')
        np.save(depth_path, depth_frames)
        print(f"Saved depth values to {depth_path}")

        cv2.destroyAllWindows()
